# ============================================================
# agent.py — Autonomous AI Code Agent
# Given only an instruction, the agent:
#   1. Uses RAG to find relevant codebase context
#   2. Plans which files to modify and/or create
#   3. Generates/modifies each file
#   4. Validates build after each change
#   5. Self-corrects on compilation errors
# ============================================================

import os
import sys
import json
import shutil
import argparse
import requests
import chromadb
from datetime import datetime
from config import (
    OLLAMA_GENERATE_URL, OLLAMA_CODING_MODEL,
    OLLAMA_EMBED_URL, OLLAMA_EMBED_MODEL,
    CHROMA_HOST, CHROMA_PORT,
    CHROMA_TENANT, CHROMA_DATABASE,
    CHROMA_JAVA_COLLECTION, CHROMA_ANGULAR_COLLECTION,
    MAX_CORRECTION_LOOPS, RAG_TOP_K,
    MAX_CHUNK_CHARS, LOG_FILE,
    JAVA_REPO_ROOT, ANGULAR_REPO_ROOT
)
from build_validator import validate_file, format_errors_for_llm

# ---- Logging ------------------------------------------------
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

# ---- File Operations ----------------------------------------
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def write_file(path: str, content: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def backup_file(path: str):
    shutil.copy2(path, path + ".bak")
    log(f"Backup: {path}.bak")

def restore_backup(path: str):
    if os.path.exists(path + ".bak"):
        shutil.copy2(path + ".bak", path)
        log(f"Restored: {path}")

def cleanup_backup(path: str):
    bak = path + ".bak"
    if os.path.exists(bak):
        os.remove(bak)

def delete_file(path: str):
    if os.path.exists(path):
        os.remove(path)
        log(f"Deleted failed new file: {path}")

# ---- Language Detection -------------------------------------
def detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".java":
        return "java"
    elif ext in [".ts", ".html", ".scss", ".css"]:
        return "angular"
    return "unknown"

def language_hint(file_path: str) -> str:
    lang = detect_language(file_path)
    return "Java Spring Boot" if lang == "java" else "Angular TypeScript" if lang == "angular" else "code"

def infer_java_package(file_path: str) -> str:
    try:
        normalized = file_path.replace("\\", "/")
        idx = normalized.find("/java/")
        if idx == -1:
            return ""
        rel = normalized[idx + 6:]
        return ".".join(rel.split("/")[:-1])
    except Exception:
        return ""

# ---- ChromaDB Client ----------------------------------------
def get_chroma_client():
    return chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        tenant=CHROMA_TENANT,
        database=CHROMA_DATABASE
    )

# ---- Embedding ----------------------------------------------
def get_embedding(text: str) -> list:
    try:
        response = requests.post(
            OLLAMA_EMBED_URL,
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text[:MAX_CHUNK_CHARS]},
            timeout=120
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        log(f"WARNING: Embedding failed: {e}")
        return None

# ---- RAG: Retrieve Context ----------------------------------
def retrieve_context(query: str, language: str, exclude_file: str = None) -> list:
    """
    Returns list of dicts: [{file_path, relative_path, content}]
    """
    log(f"RAG query: {query[:80]}...")
    try:
        client = get_chroma_client()
        collection_name = CHROMA_JAVA_COLLECTION if language == "java" else CHROMA_ANGULAR_COLLECTION
        collection = client.get_collection(collection_name)

        embedding = get_embedding(query)
        if embedding:
            results = collection.query(
                query_embeddings=[embedding],
                n_results=RAG_TOP_K,
                include=["documents", "metadatas"]
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=RAG_TOP_K,
                include=["documents", "metadatas"]
            )

        context_files = []
        seen_paths = set()
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            fp = meta.get("file_path", "")
            if exclude_file and os.path.basename(fp) == os.path.basename(exclude_file):
                continue
            if fp in seen_paths:
                continue
            seen_paths.add(fp)
            context_files.append({
                "file_path": fp,
                "relative_path": meta.get("relative_path", fp),
                "content": doc
            })

        log(f"  Retrieved {len(context_files)} context files")
        return context_files

    except Exception as e:
        log(f"WARNING: RAG failed: {e}")
        return []

def format_context(context_files: list) -> str:
    parts = []
    for cf in context_files:
        parts.append(f"// File: {cf['relative_path']}\n{cf['content']}")
    return "\n\n".join(parts)

# ---- Ollama -------------------------------------------------
def call_ollama(prompt: str, temperature: float = 0.1) -> str:
    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": OLLAMA_CODING_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature, "num_predict": 4096}
            },
            timeout=300
        )
        response.raise_for_status()
        return response.json()["response"]
    except requests.exceptions.Timeout:
        log("ERROR: Ollama timed out")
        return None
    except Exception as e:
        log(f"ERROR: Ollama call failed: {e}")
        return None

def clean_code(response: str) -> str:
    if not response:
        return response
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    return response.strip()

def clean_json(response: str) -> str:
    """Strip markdown json fences if present."""
    if not response:
        return response
    response = response.strip()
    if response.startswith("```"):
        lines = response.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        response = "\n".join(lines)
    return response.strip()

# ============================================================
# PHASE 1 — PLANNING
# Agent decides which files to modify and which to create
# ============================================================

PLAN_PROMPT = """You are an expert software architect reviewing a codebase.
Based on the instruction and codebase context provided, create a detailed plan of files to modify and new files to create.

INSTRUCTION:
{instruction}

EXISTING CODEBASE CONTEXT:
{context}

JAVA REPO ROOT: {java_root}
ANGULAR REPO ROOT: {angular_root}

Respond ONLY with a valid JSON object in this exact format (no explanation, no markdown):
{{
  "summary": "brief description of what will be done",
  "modify": [
    {{
      "file_path": "absolute file path to modify",
      "reason": "why this file needs to change",
      "changes": "specific changes to make"
    }}
  ],
  "create": [
    {{
      "file_path": "absolute file path for new file",
      "reason": "why this file is needed",
      "description": "what this file should contain"
    }}
  ]
}}

RULES:
- Use absolute file paths based on the repo roots provided
- Only include files that genuinely need to change for this instruction
- For Java files: follow existing package structure
- For Angular files: follow existing module/component structure
- If no files need modification, use empty array []
- If no new files are needed, use empty array []"""


def plan_changes(instruction: str) -> dict:
    """
    Ask LLM to plan which files to modify and create.
    Returns parsed plan dict or None on failure.
    """
    log("\n📋 PHASE 1: Planning changes...")

    # Retrieve context from both collections
    java_context = retrieve_context(instruction, "java")
    angular_context = retrieve_context(instruction, "angular")
    all_context = format_context(java_context + angular_context)

    prompt = PLAN_PROMPT.format(
        instruction=instruction,
        context=all_context if all_context else "No existing context found.",
        java_root=JAVA_REPO_ROOT,
        angular_root=ANGULAR_REPO_ROOT
    )

    log("Asking LLM to plan file changes...")
    response = call_ollama(prompt, temperature=0.1)

    if not response:
        log("ERROR: No planning response from LLM")
        return None

    # Parse JSON plan
    try:
        cleaned = clean_json(response)
        plan = json.loads(cleaned)
        log(f"\n Plan Summary: {plan.get('summary', 'N/A')}")
        log(f"  Files to modify : {len(plan.get('modify', []))}")
        log(f"  Files to create : {len(plan.get('create', []))}")

        for item in plan.get("modify", []):
            log(f"  ✏️  MODIFY: {item['file_path']}")
            log(f"      Reason: {item['reason']}")

        for item in plan.get("create", []):
            log(f"  ➕ CREATE: {item['file_path']}")
            log(f"      Reason: {item['reason']}")

        return plan

    except json.JSONDecodeError as e:
        log(f"ERROR: Could not parse plan JSON: {e}")
        log(f"Raw response:\n{response[:500]}")
        return None


# ============================================================
# PHASE 2 — CODE GENERATION
# ============================================================

def build_modify_prompt(instruction: str, file_path: str, code: str, context: str, specific_changes: str) -> str:
    return f"""You are an expert {language_hint(file_path)} developer.
Modify the file below based on the instruction.

INSTRUCTION:
{instruction}

SPECIFIC CHANGES NEEDED:
{specific_changes}

RELATED CODEBASE CONTEXT:
{context}

FILE TO MODIFY: {os.path.basename(file_path)}
```
{code}
```

RULES:
- Return ONLY the complete updated file content
- No explanations, no markdown, no code fences
- Preserve all existing functionality unless instructed
- Include all necessary imports
- Follow {language_hint(file_path)} best practices"""


def build_create_prompt(instruction: str, file_path: str, context: str, description: str) -> str:
    package_hint = ""
    if detect_language(file_path) == "java":
        pkg = infer_java_package(file_path)
        if pkg:
            package_hint = f"Java package: {pkg}"

    return f"""You are an expert {language_hint(file_path)} developer.
Create a brand new file based on the instruction below.

INSTRUCTION:
{instruction}

FILE TO CREATE: {os.path.basename(file_path)}
{package_hint}

DESCRIPTION OF WHAT THIS FILE SHOULD DO:
{description}

RELATED CODEBASE CONTEXT (follow these patterns and conventions):
{context}

RULES:
- Return ONLY the complete file content, ready to compile
- No explanations, no markdown, no code fences
- Follow naming conventions from the codebase context
- Include all necessary imports
- Follow {language_hint(file_path)} best practices
- Must compile without errors"""


def build_correction_prompt(instruction: str, file_path: str, code: str, errors: str, attempt: int) -> str:
    return f"""You are an expert developer fixing compilation errors in {os.path.basename(file_path)}.

ORIGINAL INSTRUCTION:
{instruction}

CURRENT CODE:
```
{code}
```

COMPILATION ERRORS (attempt {attempt}/{MAX_CORRECTION_LOOPS}):
{errors}

RULES:
- Fix ALL errors listed above
- Keep the original intent intact
- Return ONLY the complete corrected file
- No explanations, no markdown, no code fences"""


# ============================================================
# PHASE 3 — EXECUTE + VALIDATE + SELF-CORRECT
# ============================================================

def process_single_file(
    file_path: str,
    instruction: str,
    mode: str,           # "modify" or "create"
    specific_info: str   # changes description or file description
) -> bool:
    """
    Process one file: generate code, write, validate, self-correct.
    Returns True on success, False on failure.
    """
    log(f"\n{'─'*60}")
    log(f"{'✏️  MODIFYING' if mode == 'modify' else '➕ CREATING'}: {file_path}")
    log(f"{'─'*60}")

    language = detect_language(file_path)
    is_new = mode == "create"

    # Backup existing file
    if not is_new and os.path.exists(file_path):
        backup_file(file_path)
        original_code = read_file(file_path)
    elif not is_new and not os.path.exists(file_path):
        log(f"ERROR: File not found for modify: {file_path}")
        return False
    else:
        original_code = None

    # Get focused RAG context for this specific file
    context_files = retrieve_context(
        query=f"{instruction} {os.path.basename(file_path)}",
        language=language,
        exclude_file=file_path
    )
    context = format_context(context_files)

    # Generate code
    log("Generating code...")
    if is_new:
        prompt = build_create_prompt(instruction, file_path, context, specific_info)
    else:
        prompt = build_modify_prompt(instruction, file_path, original_code, context, specific_info)

    response = call_ollama(prompt)
    if not response:
        log("ERROR: No LLM response")
        if not is_new:
            restore_backup(file_path)
        return False

    generated_code = clean_code(response)
    write_file(file_path, generated_code)
    log(f"Written: {len(generated_code)} chars")

    # Validate build
    log("Validating build...")
    build_result = validate_file(file_path)

    if build_result.success:
        log(f"✅ Build passed!")
        if not is_new:
            cleanup_backup(file_path)
        return True

    # Self-correction loop
    log(f"Build failed — self-correcting (max {MAX_CORRECTION_LOOPS} attempts)...")
    current_code = generated_code

    for attempt in range(1, MAX_CORRECTION_LOOPS + 1):
        log(f"\n  Correction attempt {attempt}/{MAX_CORRECTION_LOOPS}")
        error_msg = format_errors_for_llm(build_result, file_path)

        correction_prompt = build_correction_prompt(
            instruction, file_path, current_code, error_msg, attempt
        )

        corrected_response = call_ollama(correction_prompt)
        if not corrected_response:
            continue

        current_code = clean_code(corrected_response)
        write_file(file_path, current_code)
        build_result = validate_file(file_path)

        if build_result.success:
            log(f"✅ Fixed after {attempt} correction(s)!")
            if not is_new:
                cleanup_backup(file_path)
            return True

        log(f"  Still failing...")

    # All attempts failed
    log(f"❌ Failed after {MAX_CORRECTION_LOOPS} attempts")
    if is_new:
        delete_file(file_path)
    else:
        restore_backup(file_path)
    return False


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def run_agent(instruction: str) -> bool:
    log(f"\n{'='*60}")
    log(f"AUTONOMOUS AGENT STARTED")
    log(f"Instruction: {instruction}")
    log(f"{'='*60}")

    # Phase 1: Plan
    plan = plan_changes(instruction)
    if not plan:
        log("ERROR: Planning failed — cannot proceed")
        return False

    total = len(plan.get("modify", [])) + len(plan.get("create", []))
    if total == 0:
        log("ℹ️  No file changes identified for this instruction")
        return True

    # Phase 2 & 3: Execute plan
    log(f"\n⚙️  PHASE 2: Executing plan ({total} file(s))...")

    results = []

    # Process modifications first (existing code as context for new files)
    for item in plan.get("modify", []):
        success = process_single_file(
            file_path=item["file_path"],
            instruction=instruction,
            mode="modify",
            specific_info=item.get("changes", item.get("reason", ""))
        )
        results.append({"file": item["file_path"], "mode": "modify", "success": success})

    # Then process new file creations
    for item in plan.get("create", []):
        success = process_single_file(
            file_path=item["file_path"],
            instruction=instruction,
            mode="create",
            specific_info=item.get("description", item.get("reason", ""))
        )
        results.append({"file": item["file_path"], "mode": "create", "success": success})

    # Summary
    log(f"\n{'='*60}")
    log(f"AGENT EXECUTION SUMMARY")
    log(f"{'='*60}")
    success_count = sum(1 for r in results if r["success"])
    fail_count = len(results) - success_count

    for r in results:
        icon = "✅" if r["success"] else "❌"
        mode_label = "MODIFIED" if r["mode"] == "modify" else "CREATED"
        log(f"{icon} [{mode_label}] {r['file']}")

    log(f"\nTotal: {len(results)} | ✅ {success_count} succeeded | ❌ {fail_count} failed")

    return fail_count == 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous AI Code Agent")
    parser.add_argument(
        "--instruction",
        required=True,
        help="What functionality to implement or change"
    )
    args = parser.parse_args()

    success = run_agent(instruction=args.instruction)

    if success:
        log("\n🎉 Agent completed successfully!")
        sys.exit(0)
    else:
        log("\n💥 Agent finished with failures — see log for details")
        sys.exit(1)
