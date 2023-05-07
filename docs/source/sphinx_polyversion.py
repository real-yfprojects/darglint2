"""
Build sphinx docs in multiple versions.

The versions are extracted from git, build and merged into one directory.
"""
import argparse
import enum
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime
from pathlib import Path, PurePath
from typing import Iterable, List, NamedTuple, Optional, Union, cast

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
    def default(self, o):
        if isinstance(o, GitRef):
            return o._asdict()
        if isinstance(o, GitRefType):
            return str(GitRefType)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)


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


def get_git_root(directory: Path):
    cmd = (
        "git",
        "rev-parse",
        "--show-toplevel",
    )
    return Path(subprocess.check_output(cmd, cwd=directory).decode().rstrip("\n"))


regex_ref = r"refs/(?P<type>heads|tags|remotes/(?P<remote>[^/]+))/(?P<name>\S+)"
pattern_ref = re.compile(regex_ref)


def get_all_refs(repo: Path):
    cmd = (
        "git",
        "for-each-ref",
        "--format",
        "%(objectname)\t%(refname)\t%(creatordate:iso)",
        "refs",
    )
    lines = subprocess.check_output(cmd).decode().splitlines()
    for line in lines:
        obj, ref, date_str = line.split("\t")
        date = datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S %z")

        match = pattern_ref.fullmatch(ref)
        if not match:
            raise ValueError(f"Invalid ref {ref}")
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


def get_refs(
    repo: Path,
    branch_regex: str,
    tag_regex: str,
    remote: Optional[str] = None,
    files: Iterable[PurePath] = (),
):
    def predicate(ref: GitRef):
        match = True
        if ref.type == GitRefType.TAG:
            match = re.fullmatch(tag_regex, ref.name)
        if ref.type == GitRefType.BRANCH:
            match = re.fullmatch(branch_regex, ref.name)
        for file in files:
            if not file_exists(repo, ref, file):
                return False
        return ref.remote == remote and match

    branches: List[GitRef] = []
    tags: List[GitRef] = []
    for ref in get_all_refs(repo):
        if not predicate(ref):
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


def file_exists(repo: Path, ref: GitRef, file: PurePath):
    cmd = (
        "git",
        "cat-file",
        "-e",
        "{}:{}".format(ref.ref, file.as_posix()),
    )
    return (
        subprocess.run(
            cmd, cwd=repo, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        ).returncode
        == 0
    )


def copy_tree(repo: Path, ref: GitRef, dest: Union[str, Path], buffer_size=0):
    # retrieve commit contents as tar archive
    cmd = ("git", "archive", "--format", "tar", ref.obj)
    with tempfile.SpooledTemporaryFile(max_size=buffer_size) as f:
        subprocess.check_call(cmd, cwd=repo, stdout=f)
        # extract tar archive to dir
        f.seek(0)
        with tarfile.open(fileobj=f) as tf:
            tf.extractall(str(dest))


# -- Sphinx ------------------------------------------------------------------


def bundle_metadata(tags: List[GitRef], branches: List[GitRef]):
    return {"tags": tags, "branches": branches}


def get_version_output_dir(output_dir: Path, ref: GitRef):
    return output_dir / ref.name


def run_sphinx(
    cwd: Path, source: Path, build: Path, metadata: dict, *, sphinx_args: Iterable[str]
):
    cmd: List[str] = ["sphinx-build", str(source), str(build)]
    cmd += sphinx_args
    env = os.environ.copy()
    env["VERSION_METADATA"] = json.dumps(metadata, cls=GitRefEncoder)
    process = subprocess.run(
        cmd, cwd=cwd, env=env, stdout=sys.stdout, stderr=sys.stderr
    )
    return process


def build_version(
    repo: Path,
    rel_source: PurePath,
    output_dir: Path,
    ref: GitRef,
    metadata: dict,
    *,
    buffer_size: int = 0,
    sphinx_args=(),
):
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        # checkout object
        copy_tree(repo, ref, tmpdir_str, buffer_size=buffer_size)

        # build
        success = (
            run_sphinx(
                repo,
                tmpdir / rel_source,
                output_dir,
                metadata,
                sphinx_args=sphinx_args,
            ).returncode
            == 0
        )

    return success


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
        shutil.copytree(static_dir, build_dir, dirs_exist_ok=True)

    # generate dynamic files from jinja templates
    if template_dir.is_dir():
        env = Environment(
            loader=FileSystemLoader(str(template_dir)), autoescape=select_autoescape()
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

    return parser


def main():
    parser = get_parser()
    options, sphinx_args = parser.parse_known_args()

    # determine git root
    source_dir = cast(Path, options.source_dir)
    output_dir = cast(Path, options.output_dir)
    repo = get_git_root(source_dir)

    rel_source = source_dir.absolute().relative_to(repo)

    # determine, categorize and sort versions
    tags, branches = get_refs(
        repo, options.branch_regex, options.tag_regex, options.remote, [rel_source]
    )

    # metadata dict
    global_metadata = bundle_metadata(tags, branches)

    # make output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # run sphinx
    for version in tags + branches:
        print(f"Building {version.name}...")
        meta: dict = {"current": version}
        meta.update(global_metadata)
        success = build_version(
            repo,
            rel_source,
            output_dir / version.name,
            version,
            meta,
            buffer_size=options.buffer,
            sphinx_args=sphinx_args,
        )

        if not success:
            print(f"Failed building {version.name}.", file=sys.stderr)
            if version in tags:
                tags.remove(version)
            elif version in branches:
                branches.remove(version)

    # build local version
    if options.local:
        meta: dict = {"current": "local"}
        meta.update(global_metadata)
        run_sphinx(
            repo,
            source_dir,
            output_dir / "local",
            meta,
            sphinx_args=sphinx_args,
        )

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
    main()
