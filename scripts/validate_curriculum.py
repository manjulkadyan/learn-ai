"""Check curriculum IDs, links, prerequisite graph, and subtopic content shape."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / "curriculum.json").read_text(encoding="utf-8"))
topics = manifest["topics"]
assert manifest["schema_version"] == 1
assert topics, "curriculum must contain topics"


def unique(values, description):
    assert len(values) == len(set(values)), f"{description} must be unique"


def extract_learning_units(block, subtopic_id, page_path):
    heading = re.search(r"^####\s+Learning units\s*$", block, re.I | re.M)
    assert heading, f"{subtopic_id} in {page_path} has no Learning units section"
    units = []
    for line in block[heading.end():].splitlines():
        if line.startswith("#"):
            break
        match = re.match(r"^\s*(?:\d+[.)]|[-*])\s+\*\*(.+?)\*\*", line)
        if match:
            units.append(match.group(1).strip().rstrip(".:;"))
        elif units:
            # Only the contiguous list directly beneath this heading contains
            # learning units; numbered worked-example steps are not units.
            break
    assert units, f"{subtopic_id} in {page_path} has no titled learning units"
    return units


unique([topic["id"] for topic in topics], "topic IDs")
all_subtopics = []
topic_pages = {}

for topic in topics:
    path = ROOT / topic["path"]
    assert path.is_file(), f"missing topic page: {topic['path']}"
    topic_pages[topic["id"]] = path
    assert topic.get("subtopics"), f"topic {topic['id']} has no subtopics"
    for subtopic in topic["subtopics"]:
        assert subtopic.get("learning_units"), f"subtopic {subtopic['id']} has no learning units"
        all_subtopics.append(subtopic)

subtopic_ids = [subtopic["id"] for subtopic in all_subtopics]
unique(subtopic_ids, "subtopic IDs")
subtopic_by_id = {subtopic["id"]: subtopic for subtopic in all_subtopics}

unit_ids = []
for subtopic in all_subtopics:
    for prerequisite in subtopic.get("prerequisites", []):
        assert prerequisite in subtopic_by_id, (
            f"unknown prerequisite {prerequisite} for subtopic {subtopic['id']}"
        )
    for unit in subtopic["learning_units"]:
        if isinstance(unit, dict):
            unit_id, unit_title = unit.get("id", ""), unit.get("title", "")
        else:
            unit_id = unit_title = unit
        assert unit_id and unit_title, f"incomplete learning unit {unit!r}"
        unit_ids.append(unit_id)
    for alternatives in subtopic.get("prerequisites_any_of", []):
        assert alternatives, f"empty alternative prerequisite group for {subtopic['id']}"
        for prerequisite in alternatives:
            assert prerequisite in subtopic_by_id, (
                f"unknown alternative prerequisite {prerequisite} for subtopic {subtopic['id']}"
            )
unique(unit_ids, "learning unit IDs")

# Prerequisites describe ordering, so reject cycles before publishing the map.
visiting = set()
visited = set()


def visit(subtopic_id):
    if subtopic_id in visiting:
        raise AssertionError(f"cyclic subtopic prerequisite involving {subtopic_id}")
    if subtopic_id in visited:
        return
    visiting.add(subtopic_id)
    for prerequisite in subtopic_by_id[subtopic_id].get("prerequisites", []):
        visit(prerequisite)
    for alternatives in subtopic_by_id[subtopic_id].get("prerequisites_any_of", []):
        for prerequisite in alternatives:
            visit(prerequisite)
    visiting.remove(subtopic_id)
    visited.add(subtopic_id)


for subtopic_id in subtopic_ids:
    visit(subtopic_id)

# Alternative prerequisites are a choice among equivalent entry branches.
for subtopic in all_subtopics:
    for alternatives in subtopic.get("prerequisites_any_of", []):
        assert len(alternatives) >= 2, (
            f"alternative prerequisite group for {subtopic['id']} needs at least two choices"
        )
        assert len(alternatives) == len(set(alternatives)), (
            f"duplicate alternative prerequisite for {subtopic['id']}"
        )

# Match each manifest subtopic title to a heading on its topic page. This guards
# against the inventory silently inventing a parallel curriculum.
for topic in topics:
    content = topic_pages[topic["id"]].read_text(encoding="utf-8")
    headings = []
    for line in content.splitlines():
        match = re.match(r"^(#{2,4})\s+(.+)$", line)
        if match:
            heading = re.sub(r"^Subtopic:\s*", "", match.group(2), flags=re.I)
            heading = re.sub(r"^\d+\.\s*", "", heading)
            if heading.lower() != "learning units" and heading.lower() != "subtopics":
                headings.append(heading.strip().lower())
    normalized_headings = {re.sub(r"[^a-z0-9]+", "", heading) for heading in headings}
    for subtopic in topic["subtopics"]:
        normalized_title = re.sub(r"[^a-z0-9]+", "", subtopic["title"].lower())
        assert normalized_title in normalized_headings, (
            f"manifest subtopic {subtopic['id']} ({subtopic['title']}) "
            f"has no matching heading in {topic['path']}"
        )

    # Every declared teaching element must appear under each subtopic heading.
    lines = content.splitlines()
    for subtopic in topic["subtopics"]:
        title = re.escape(subtopic["title"].lower())
        def heading_title(line):
            title = re.sub(r"^#+\s*", "", line).strip()
            title = re.sub(r"^Subtopic:\s*", "", title, flags=re.I)
            title = re.sub(r"^\d+\.\s*", "", title)
            return re.sub(r"[^a-z0-9]+", "", title.lower())

        normalized_title = re.sub(r"[^a-z0-9]+", "", subtopic["title"].lower())
        start = next(
            i for i, line in enumerate(lines)
            if line.startswith("#") and heading_title(line) == normalized_title
        )
        heading_level = len(lines[start]) - len(lines[start].lstrip("#"))
        end = next(
            (i for i in range(start + 1, len(lines))
             if lines[i].startswith("#" * heading_level + " ")
             and (len(lines[i]) == heading_level + 1 or lines[i][heading_level + 1] != "#")),
            len(lines),
        )
        block = "\n".join(lines[start:end])
        expected_units = [
            unit.get("title", "") if isinstance(unit, dict) else unit
            for unit in subtopic["learning_units"]
        ]
        actual_units = extract_learning_units(block, subtopic["id"], topic["path"])
        assert actual_units == expected_units, (
            f"learning units for {subtopic['id']} in {topic['path']} differ in "
            f"title, count, or order; manifest={expected_units!r}, page={actual_units!r}"
        )
        required = {
            "beginner explanation": r"Beginner explanation",
            "prerequisites": r"Prerequisites",
            "concepts and vocabulary": r"Concepts and vocabulary|Learning units.*concepts and vocabulary",
            "worked example": r"Worked example",
            "illustration or animation": r"Illustration(?: or animation)?(?: cue)?",
            "code practice": r"Code practice",
            "common mistakes": r"Common mistakes",
            "exercises and solutions": r"Exercises(?: and solution guidance| and solutions)?",
            "practice project / practical artifact": r"Practice project|Practical artifact",
            "mastery checkpoint": r"Mastery checkpoint",
        }
        missing = [label for label, pattern in required.items() if not re.search(pattern, block, re.I)]
        assert not missing, f"{subtopic['id']} in {topic['path']} missing: {', '.join(missing)}"
        visual_cue = re.search(r"(?:Illustration(?: or animation)?(?: cue)?|Illustration or animation)\.?:?", block, re.I)
        assert visual_cue, f"{subtopic['id']} in {topic['path']} missing its visual guidance"
        next_field = re.search(r"\n\*\*(?:Code practice|Common mistakes|Exercises)", block[visual_cue.end():], re.I)
        visual_section = block[visual_cue.end():visual_cue.end() + next_field.start()] if next_field else block[visual_cue.end():]
        has_renderable_visual = "```mermaid" in visual_section or re.search(r"\[[^]]*\]\([^)]*\.svg(?:#[^)]*)?\)", visual_section, re.I)
        assert has_renderable_visual, (
            f"{subtopic['id']} in {topic['path']} needs a Mermaid diagram or linked illustration under its visual guidance"
        )

        # Each subtopic owns exactly one practical artifact and one mastery
        # checkpoint. Topic-level capstones use distinct headings and labels.
        projects = re.findall(r"^\*\*Practice project(?:\.|:)\*\*", block, re.I | re.M)
        checkpoints = re.findall(r"^\*\*Mastery checkpoint(?:\.|:)\*\*", block, re.I | re.M)
        assert len(projects) == 1, (
            f"{subtopic['id']} in {topic['path']} must have exactly one practice project; "
            f"found {len(projects)}"
        )
        assert len(checkpoints) == 1, (
            f"{subtopic['id']} in {topic['path']} must have exactly one mastery checkpoint; "
            f"found {len(checkpoints)}"
        )

# Mermaid blocks must be balanced and have standalone fences so diagrams render.
for page in (ROOT / "docs/topics").glob("*.md"):
    in_mermaid = False
    for number, line in enumerate(page.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip() == "```mermaid":
            assert not in_mermaid, f"nested Mermaid opener in {page}:{number}"
            in_mermaid = True
        elif line.strip() == "```" and in_mermaid:
            in_mermaid = False
        elif in_mermaid:
            assert "```" not in line, f"inline Mermaid fence in {page}:{number}"
    assert not in_mermaid, f"unclosed Mermaid block in {page}"

# Standalone lessons remain supporting material and may optionally estimate time.
# An estimate never becomes a curriculum prerequisite or mastery condition.
for page in (ROOT / "docs/lessons").rglob("*.md"):
    assert page.read_text(encoding="utf-8").strip(), f"empty lesson page: {page}"

print(
    f"Validated {len(topics)} topics, {len(all_subtopics)} subtopics, "
    f"{len(unit_ids)} learning units, required subtopic content, and acyclic prerequisites."
)
