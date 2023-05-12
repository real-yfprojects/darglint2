#!/usr/bin/env python3
"""
Build sphinx docs in multiple versions.

The versions are extracted from git, build and merged into one directory.
"""
import argparse
import asyncio
import enum
import json
import os
import re
import shutil
import sys
import tarfile
import tempfile
from asyncio.subprocess import DEVNULL, PIPE
from datetime import datetime
from pathlib import Path, PurePath
from subprocess import CalledProcessError
from typing import Any, Iterable, List, NamedTuple, Optional, Union, cast

from jinja2 import Environment, FileSystemLoader, select_autoescape

# -- Configuration -----------------------------------------------------------

BRANCH_REGEX = r".*docs.*|master"
TAG_REGEX = r".*"

# -- Git ---------------------------------------------------------------------


class GitRefType(enum.Enum):
    TAG = enum.auto()
    BRANCH = enum.auto()


class GitRef(NamedTuple):
    name: str
    obj: str  # hash
    ref: str  # git ref
    type: GitRefType
    date: datetime  # creation
    remote: Optional[str] = None


class GitRefEncoder(json.JSONEncoder):
    def replace(self, o):
        if isinstance(o, GitRef):
            return o._asdict()
        if isinstance(o, GitRefType):
            return o.name
        if isinstance(o, datetime):
            return o.isoformat()

    def replace_types(self, o):
        replacement = self.replace(o)
        if replacement:
            return self.replace_types(replacement)
        if isinstance(o, dict):
            return {k: self.replace_types(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [self.replace_types(v) for v in o]
        return o

    def iterencode(self, o: Any, _one_shot: bool = False):
        return super().iterencode(self.replace_types(o), _one_shot)


class GitRefDecoder(json.JSONDecoder):
    def __init__(self, **kwargs) -> None:
        super().__init__(object_hook=self.object_hook, **kwargs)

    def object_hook(self, object: dict):
        object_attr = set(object.keys())
        default_attr = set(GitRef._field_defaults.keys())
        all_attr = set(GitRef._fields)
        if object_attr <= all_attr and (all_attr - default_attr) <= object_attr:
            t = GitRefType[object["type"]]
            d = datetime.fromisoformat(object["date"])
            object.update({"date": d, "type": t})
            return GitRef(**object)
        return object


async def get_git_root(directory: Path):
    cmd = (
        "git",
        "rev-parse",
        "--show-toplevel",
    )
    process = await asyncio.create_subprocess_exec(*cmd, cwd=directory, stdout=PIPE)
    out, err = await process.communicate()
    return Path(out.decode().rstrip("\n"))


regex_ref = r"refs/(?P<type>heads|tags|remotes/(?P<remote>[^/]+))/(?P<name>\S+)"
pattern_ref = re.compile(regex_ref)


async def get_all_refs(repo: Path):
    cmd = (
        "git",
        "for-each-ref",
        "--format",
        "%(objectname)\t%(refname)\t%(creatordate:iso)",
        "refs",
    )
    process = await asyncio.create_subprocess_exec(*cmd, stdout=PIPE)
    out, err = await process.communicate()
    lines = out.decode().splitlines()
    for line in lines:
        obj, ref, date_str = line.split("\t")
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")

        match = pattern_ref.fullmatch(ref)
        if not match:
            print(f"Invalid ref {ref}", file=sys.stderr)
            continue
        name = match["name"]
        type_str = match["type"]
        remote = None
        if type_str == "heads":
            type_ = GitRefType.BRANCH
        elif type_str == "tags":
            type_ = GitRefType.TAG
        elif type_str.startswith("remotes"):
            type_ = GitRefType.BRANCH
            remote = match["remote"]
        else:
            raise ValueError(f"Unknown type {type_str}")

        yield GitRef(name, obj, ref, type_, date, remote)


async def get_refs(
    repo: Path,
    branch_regex: str,
    tag_regex: str,
    remote: Optional[str] = None,
    files: Iterable[PurePath] = (),
):
    async def predicate(ref: GitRef):
        match = True
        if ref.type == GitRefType.TAG:
            match = re.fullmatch(tag_regex, ref.name)
        if ref.type == GitRefType.BRANCH:
            match = re.fullmatch(branch_regex, ref.name)
        for file in files:
            if not (await file_exists(repo, ref, file)):
                return False
        return ref.remote == remote and match

    branches: List[GitRef] = []
    tags: List[GitRef] = []
    async for ref in get_all_refs(repo):
        if not (await predicate(ref)):
            continue
        if ref.type == GitRefType.TAG:
            tags.append(ref)
        elif ref.type == GitRefType.BRANCH:
            branches.append(ref)

    def key(ref: GitRef):
        return ref.date

    tags = sorted(tags, key=key)
    branches = sorted(branches, key=key)

    return tags, branches


async def file_exists(repo: Path, ref: GitRef, file: PurePath):
    cmd = (
        "git",
        "cat-file",
        "-e",
        "{}:{}".format(ref.ref, file.as_posix()),
    )
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=repo,
        stdout=DEVNULL,
        stderr=DEVNULL,
    )
    return (await process.wait()) == 0


async def copy_tree(repo: Path, ref: GitRef, dest: Union[str, Path], buffer_size=0):
    # retrieve commit contents as tar archive
    cmd = ("git", "archive", "--format", "tar", ref.obj)
    with tempfile.SpooledTemporaryFile(max_size=buffer_size) as f:
        process = await asyncio.create_subprocess_exec(*cmd, cwd=repo, stdout=f)
        rc = await process.wait()
        if rc != 0:
            raise CalledProcessError(f"Git archive returned {rc}")
        # extract tar archive to dir
        f.seek(0)
        with tarfile.open(fileobj=f) as tf:
            tf.extractall(str(dest))


# -- Sphinx ------------------------------------------------------------------


def bundle_metadata(tags: List[GitRef], branches: List[GitRef]):
    return {"tags": tags, "branches": branches}


def get_version_output_dir(output_dir: Path, ref: GitRef):
    return output_dir / ref.name


async def create_poetry_env(project: Path, *, poetry_args: Iterable[str]):
    cmd: List[str] = ["poetry", "install"]
    cmd += poetry_args
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # unset poetry env
    process = await asyncio.create_subprocess_exec(
        *cmd, cwd=project, env=env, stdout=PIPE, stderr=PIPE
    )
    out, err = await process.communicate()
    sys.stdout.write(out.decode())
    sys.stderr.write(err.decode())
    return process.returncode


async def run_sphinx(
    source: Path, build: Path, metadata: dict, *, sphinx_args: Iterable[str]
):
    cmd: List[str] = [
        "poetry",
        "run",
        "sphinx-build",
        "--color",
        str(source.absolute()),
        str(build.absolute()),
    ]
    cmd += sphinx_args
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)  # unset poetry env
    env["POLYVERSION_DATA"] = json.dumps(metadata, cls=GitRefEncoder)
    env["POLYVERSION_PATH"] = str(Path(__file__).absolute().resolve().parent)
    process = await asyncio.create_subprocess_exec(
        *cmd, cwd=source, env=env, stdout=PIPE, stderr=PIPE
    )
    out, err = await process.communicate()
    sys.stdout.write(out.decode())
    sys.stderr.write(err.decode())
    return process.returncode


async def build_version(
    repo: Path,
    rel_source: PurePath,
    output_dir: Path,
    ref: GitRef,
    metadata: dict,
    *,
    buffer_size: int = 0,
    sphinx_args=(),
    poetry_args=(),
):
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # checkout object
        await copy_tree(repo, ref, tmpdir_str, buffer_size=buffer_size)

        # build
        await create_poetry_env(tmpdir, poetry_args=poetry_args)
        rc = await run_sphinx(
            tmpdir / rel_source,
            output_dir,
            metadata,
            sphinx_args=sphinx_args,
        )

    return rc == 0


def shift_path(
    src_anchor: Path,
    dst_anchor: Path,
    src: Path,
):
    return dst_anchor / src.relative_to(src_anchor)


def generate_root_dir(
    repo: Path,
    source_dir: Path,
    build_dir: Path,
    metadata: dict,
    static_dir: Path,
    template_dir: Path,
):
    # metadata as json
    (build_dir / "versions.json").write_text(json.dumps(metadata, cls=GitRefEncoder))

    # copy static files
    if static_dir.exists():
        for file in static_dir.rglob("*"):
            shutil.copyfile(file, shift_path(static_dir, build_dir, file))

    # generate dynamic files from jinja templates
    if template_dir.is_dir():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=select_autoescape(),
        )
        for template_path_str in env.list_templates():
            template = env.get_template(template_path_str)
            rendered = template.render(**metadata, repo=repo)
            output_path = build_dir / template_path_str
            output_path.write_text(rendered)


# -- Main --------------------------------------------------------------------


def get_parser():
    parser = argparse.ArgumentParser(
        "sphinx_polyversion.py",
        description="Build sphinx docs from multiple versions.",
    )

    parser.add_argument("source_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("-r", "--remote", default=None)
    parser.add_argument("-t", "--tag_regex", default=TAG_REGEX)
    parser.add_argument("-b", "--branch_regex", default=BRANCH_REGEX)
    parser.add_argument("-l", "--local", action="store_true")
    parser.add_argument("--buffer", default=10**8)
    parser.add_argument("--template_dir", type=Path)
    parser.add_argument("--static_dir", type=Path)
    parser.add_argument("--poetry-groups", nargs="*")

    return parser


async def identity(v):
    return v


async def main():
    parser = get_parser()
    options, sphinx_args = parser.parse_known_args()

    # determine git root
    source_dir = cast(Path, options.source_dir)
    output_dir = cast(Path, options.output_dir)
    repo = await get_git_root(source_dir)

    rel_source = source_dir.absolute().relative_to(repo)

    # determine, categorize and sort versions
    tags, branches = await get_refs(
        repo, options.branch_regex, options.tag_regex, options.remote, [rel_source]
    )

    # metadata dict
    global_metadata = bundle_metadata(tags, branches)

    # make output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # poetry args used for building
    poetry_args = ["--only", ",".join(options.poetry_groups)]

    # build local version
    local_build = identity(True)
    if options.local:
        meta: dict = {
            "current": GitRef("local", "None", "None", GitRefType.TAG, datetime.now())
        }
        meta.update(global_metadata)
        local_build = run_sphinx(
            source_dir,
            output_dir / "local",
            meta,
            sphinx_args=sphinx_args,
            poetry_args=poetry_args,
        )

    # run sphinx
    build_coroutines = {}
    for version in tags + branches:
        print(f"Building {version.name}...")
        meta: dict = {"current": version}
        meta.update(global_metadata)
        build_coroutines[version] = build_version(
            repo,
            rel_source,
            output_dir / version.name,
            version,
            meta,
            buffer_size=options.buffer,
            sphinx_args=sphinx_args,
            poetry_args=poetry_args,
        )

    local_build_success, *results = await asyncio.gather(
        local_build, *build_coroutines.values()
    )

    for version, success in zip(build_coroutines.keys(), results):
        if not success:
            print(f"Failed building {version.name}.", file=sys.stderr)
            if version in tags:
                tags.remove(version)
            elif version in branches:
                branches.remove(version)

    if not (local_build_success or branches or tags):
        raise RuntimeError("No version of the docs could be build successfully.")

    # root dir
    generate_root_dir(
        repo,
        source_dir,
        output_dir,
        global_metadata,
        options.static_dir or source_dir / "_polyversion/static",
        options.template_dir or source_dir / "_polyversion/templates",
    )


if __name__ == "__main__":
    asyncio.run(main())
