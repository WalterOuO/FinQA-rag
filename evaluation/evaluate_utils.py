import json

def calculate_reciprocal_rank(sources, source_file):
  for rank, source in enumerate(sources, start=1):
    if source.get("file_name") == source_file:
      return 1 / rank

  return 0.0


def calculate_hit_at_k(sources, source_file, k):
  return any(
    source.get("file_name") == source_file
    for source in sources[:k]
    )


def parse_judge_response(content):
  content = content.strip()

  try:
    return json.loads(content)
  except json.JSONDecodeError:
    start = content.find("{")
    end = content.rfind("}")

    if start != -1 and end != -1:
        return json.loads(content[start:end + 1])

    raise ValueError(f"Invalid Judge JSON: {content}")