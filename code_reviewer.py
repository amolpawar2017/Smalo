# ============================================================
# code_reviewer.py — AI-powered Code Review
# Identifies performance blockers, anti-patterns, security
# issues and code quality problems across Java + Angular
# ============================================================

import os
import json
import requests
from datetime import datetime
from config import (
    OLLAMA_GENERATE_URL, OLLAMA_CODING_MODEL,
    CHROMA_HOST, CHROMA_PORT,
    CHROMA_TENANT, CHROMA_DATABASE,
    CHROMA_JAVA_COLLECTION, CHROMA_ANGULAR_COLLECTION,
    OLLAMA_EMBED_URL, OLLAMA_EMBED_MODEL,
    RAG_TOP_K, MAX_CHUNK_CHARS, LOG_FILE,
    JAVA_REPO_ROOT, ANGULAR_REPO_ROOT,
    JAVA_EXTENSIONS, ANGULAR_EXTENSIONS,
    EXCLUDE_DIRS
)
import chromadb

# ---- Logging ------------------------------------------------
def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")

def log_thinking(thinking: str):
    if not thinking or not thinking.strip():
        return
    border = "─" * 60
    print(f"\n\033[36m{border}")
    print(f"  🧠 MODEL REASONING")
    print(f"{border}\033[0m")
    for line in thinking.strip().splitlines():
        print(f"\033[36m  {line}\033[0m")
    print(f"\033[36m{border}\033[0m\n")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"\n{'─'*60}\n  MODEL REASONING\n{'─'*60}\n")
        f.write(thinking.strip() + "\n")
        f.write(f"{'─'*60}\n\n")

# ---- Severity Levels ----------------------------------------
SEVERITY_CRITICAL = "🔴 CRITICAL"
SEVERITY_HIGH     = "🟠 HIGH"
SEVERITY_MEDIUM   = "🟡 MEDIUM"
SEVERITY_LOW      = "🟢 LOW"
SEVERITY_INFO     = "ℹ️  INFO"

SEVERITY_ORDER = {
    "CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4
}

# ---- Review Categories --------------------------------------
REVIEW_CATEGORIES = {
    "performance": "Performance Blockers",
    "security":    "Security Vulnerabilities",
    "memory":      "Memory Leaks",
    "concurrency": "Concurrency Issues",
    "design":      "Design Anti-Patterns",
    "quality":     "Code Quality",
    "all":         "Full Review"
}

# ---- File Detection -----------------------------------------
def detect_language(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".java":
        return "java"
    elif ext in [".ts", ".html", ".scss", ".css"]:
        return "angular"
    return "unknown"

def should_exclude(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    return any(excl in parts for excl in EXCLUDE_DIRS)

def crawl_files(root_path: str, extensions: list) -> list:
    found = []
    for dirpath, dirnames, filenames in os.walk(root_path):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS]
        if should_exclude(dirpath):
            continue
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                found.append(os.path.join(dirpath, filename))
    return found

def read_file(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""

# ---- Ollama Call --------------------------------------------
def call_ollama(prompt: str, label: str = "") -> tuple:
    """Returns (thinking, response)."""
    if label:
        log(f"🤖 {label}")

    try:
        response = requests.post(
            OLLAMA_GENERATE_URL,
            json={
                "model": OLLAMA_CODING_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 4096}
            },
            timeout=300
        )
        response.raise_for_status()
        raw = response.json().get("response", "").strip()

        # Parse <thinking> tags
        if "<thinking>" in raw and "</thinking>" in raw:
            start   = raw.find("<thinking>") + len("<thinking>")
            end     = raw.find("</thinking>")
            thinking = raw[start:end].strip()
            raw      = raw[end + len("</thinking>"):].strip()
            log_thinking(thinking)
            return thinking, raw

        return "", raw

    except requests.exceptions.Timeout:
        log("ERROR: Ollama timed out")
        return "", None
    except Exception as e:
        log(f"ERROR: {e}")
        return "", None

def clean_json(response: str) -> str:
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
# REVIEW PROMPTS
# ============================================================

JAVA_REVIEW_PROMPT = """You are a senior Java/Spring Boot performance and code quality expert.
Review the following Java file for issues.

<thinking>
Carefully analyze this code for:
1. N+1 query problems (JPA/Hibernate lazy loading in loops)
2. Missing database indexes that would cause full table scans
3. Synchronous blocking calls that should be async/reactive
4. Missing pagination on large data queries
5. Inefficient collection operations (nested loops, repeated DB calls)
6. Missing caching where data is repeatedly fetched
7. Thread safety issues and race conditions
8. Resource leaks (unclosed streams, connections)
9. Excessive object creation in hot paths
10. Missing transaction boundaries or overly long transactions
11. Security issues (SQL injection, unvalidated input, exposed secrets)
12. Spring anti-patterns (too many responsibilities, missing interfaces)
13. Exception handling gaps
14. Missing null checks leading to potential NPEs
</thinking>

After your analysis, respond ONLY with a JSON array of findings:
[
  {{
    "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
    "category": "performance|security|memory|concurrency|design|quality",
    "line_hint": "approximate line number or method name",
    "issue": "brief issue title",
    "description": "detailed explanation of the problem",
    "impact": "what will happen if not fixed",
    "recommendation": "specific fix with code example if possible"
  }}
]

FILE: {filename}
LANGUAGE: Java / Spring Boot

CODE:
```java
{code}
```

Return ONLY the JSON array. No other text."""


ANGULAR_REVIEW_PROMPT = """You are a senior Angular/TypeScript performance and code quality expert.
Review the following Angular file for issues.

<thinking>
Carefully analyze this code for:
1. Missing OnPush change detection (causes unnecessary re-renders)
2. Observable subscriptions not unsubscribed (memory leaks)
3. HTTP calls inside loops or repeated unnecessarily
4. Large synchronous computations in templates
5. Missing trackBy in *ngFor (causes full DOM re-renders)
6. Nested subscriptions instead of switchMap/mergeMap
7. Direct DOM manipulation instead of Angular patterns
8. Missing async pipe (manual subscription management)
9. Overly large components (too many responsibilities)
10. Missing error handling in HTTP calls
11. Hardcoded values that should be constants/config
12. Missing loading/error states for async operations
13. XSS vulnerabilities via innerHTML or bypassSecurityTrust
14. Missing lazy loading for feature modules
</thinking>

After your analysis, respond ONLY with a JSON array of findings:
[
  {{
    "severity": "CRITICAL|HIGH|MEDIUM|LOW|INFO",
    "category": "performance|security|memory|concurrency|design|quality",
    "line_hint": "approximate line number or method/component name",
    "issue": "brief issue title",
    "description": "detailed explanation of the problem",
    "impact": "what will happen if not fixed",
    "recommendation": "specific fix with code example if possible"
  }}
]

FILE: {filename}
LANGUAGE: Angular / TypeScript

CODE:
```typescript
{code}
```

Return ONLY the JSON array. No other text."""


# ============================================================
# REVIEW EXECUTION
# ============================================================

def review_file(file_path: str, categories: list = ["all"]) -> list:
    """
    Review a single file and return list of finding dicts.
    """
    language = detect_language(file_path)
    filename  = os.path.basename(file_path)
    code      = read_file(file_path)

    if not code.strip():
        log(f"  Skipping empty file: {filename}")
        return []

    # Truncate very large files
    if len(code) > MAX_CHUNK_CHARS * 2:
        code = code[:MAX_CHUNK_CHARS * 2]
        log(f"  File truncated for review: {filename}")

    log(f"  Reviewing: {filename}")

    if language == "java":
        prompt = JAVA_REVIEW_PROMPT.format(filename=filename, code=code)
    elif language == "angular":
        prompt = ANGULAR_REVIEW_PROMPT.format(filename=filename, code=code)
    else:
        log(f"  Skipping unsupported file type: {filename}")
        return []

    _, response = call_ollama(prompt, label=f"Reviewing {filename}")
    if not response:
        log(f"  ERROR: No response for {filename}")
        return []

    try:
        findings = json.loads(clean_json(response))
        if not isinstance(findings, list):
            findings = []

        # Filter by requested categories
        if "all" not in categories:
            findings = [f for f in findings if f.get("category") in categories]

        # Attach file info
        for f in findings:
            f["file_path"] = file_path
            f["filename"]  = filename

        log(f"  Found {len(findings)} issue(s) in {filename}")
        return findings

    except json.JSONDecodeError:
        log(f"  ERROR: Could not parse review JSON for {filename}")
        return []


# ============================================================
# REPORT GENERATION
# ============================================================

def severity_icon(severity: str) -> str:
    icons = {
        "CRITICAL": "🔴", "HIGH": "🟠",
        "MEDIUM":   "🟡", "LOW":  "🟢", "INFO": "ℹ️ "
    }
    return icons.get(severity.upper(), "⚪")

def print_report(all_findings: list, output_file: str = None):
    """Print review report to console and optionally to file."""

    # Sort by severity then file
    sorted_findings = sorted(
        all_findings,
        key=lambda f: (
            SEVERITY_ORDER.get(f.get("severity", "INFO").upper(), 99),
            f.get("filename", "")
        )
    )

    # Group by severity
    by_severity = {}
    for f in sorted_findings:
        sev = f.get("severity", "INFO").upper()
        by_severity.setdefault(sev, []).append(f)

    border = "═" * 70
    thin   = "─" * 70

    lines = []
    lines.append(f"\n{border}")
    lines.append(f"  📋 CODE REVIEW REPORT")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"{border}")

    # Summary counts
    lines.append(f"\n  SUMMARY:")
    total = len(sorted_findings)
    for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        count = len(by_severity.get(sev, []))
        if count > 0:
            lines.append(f"  {severity_icon(sev)} {sev:<10}: {count}")
    lines.append(f"  {'─'*30}")
    lines.append(f"  TOTAL      : {total} issue(s)")

    if total == 0:
        lines.append(f"\n  ✅ No issues found — codebase looks healthy!")
        lines.append(f"{border}\n")
    else:
        # Detailed findings grouped by severity
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            findings = by_severity.get(sev, [])
            if not findings:
                continue

            lines.append(f"\n{thin}")
            lines.append(f"  {severity_icon(sev)} {sev} ISSUES ({len(findings)})")
            lines.append(f"{thin}")

            for i, f in enumerate(findings, 1):
                lines.append(f"\n  [{i}] {f.get('issue', 'Unknown Issue')}")
                lines.append(f"  File     : {f.get('filename', 'N/A')}  "
                             f"(@ {f.get('line_hint', 'N/A')})")
                lines.append(f"  Category : {f.get('category', 'N/A').upper()}")
                lines.append(f"  Problem  : {f.get('description', '')}")
                lines.append(f"  Impact   : {f.get('impact', '')}")
                lines.append(f"  Fix      : {f.get('recommendation', '')}")

        lines.append(f"\n{border}\n")

    report_text = "\n".join(lines)

    # Print to console with colors
    for line in lines:
        if "CRITICAL" in line:
            print(f"\033[31m{line}\033[0m")
        elif "HIGH" in line:
            print(f"\033[33m{line}\033[0m")
        elif "MEDIUM" in line:
            print(f"\033[33m{line}\033[0m")
        elif "✅" in line:
            print(f"\033[32m{line}\033[0m")
        else:
            print(line)

    # Write to file
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(report_text)
        log(f"\n📄 Report saved: {output_file}")

    # Also log to agent log
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(report_text)

    return sorted_findings


# ============================================================
# FIX INTEGRATION — apply fixes for confirmed findings
# ============================================================

FIX_PROMPT = """You are an expert {lang} developer.
Fix the following issues in the provided code file.

<thinking>
For each issue:
1. Understand exactly what the problem is
2. Find the exact location in the code
3. Apply the minimal correct fix
4. Ensure fix doesn't break existing functionality
</thinking>

ISSUES TO FIX:
{issues_text}

FILE: {filename}
CURRENT CODE:
```
{code}
```

Return ONLY the complete fixed file content. No markdown. No explanations."""


def fix_findings(file_path: str, findings: list) -> bool:
    """
    Apply fixes for selected findings in a file.
    Returns True if fix was successful.
    """
    from build_validator import validate_file, format_errors_for_llm
    from agent import backup_file, restore_backup, cleanup_backup, write_file, MAX_CORRECTION_LOOPS

    if not findings:
        return True

    filename = os.path.basename(file_path)
    language = detect_language(file_path)
    lang_hint = "Java Spring Boot" if language == "java" else "Angular TypeScript"
    code = read_file(file_path)

    issues_text = "\n".join([
        f"{i+1}. [{f['severity']}] {f['issue']} "
        f"(@ {f.get('line_hint', 'N/A')}): {f['description']} "
        f"— Fix: {f['recommendation']}"
        for i, f in enumerate(findings)
    ])

    prompt = FIX_PROMPT.format(
        lang=lang_hint,
        issues_text=issues_text,
        filename=filename,
        code=code[:MAX_CHUNK_CHARS * 2]
    )

    backup_file(file_path)
    _, response = call_ollama(prompt, label=f"Fixing {len(findings)} issue(s) in {filename}")

    if not response:
        restore_backup(file_path)
        return False

    # Strip code fences
    fixed = response.strip()
    if fixed.startswith("```"):
        lines = fixed.split("\n")[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        fixed = "\n".join(lines).strip()

    write_file(file_path, fixed)

    # Validate build
    build_result = validate_file(file_path)
    if build_result.success:
        log(f"✅ Fix validated successfully for {filename}")
        cleanup_backup(file_path)
        return True

    # Self-correct up to MAX_CORRECTION_LOOPS times
    current = fixed
    for attempt in range(1, MAX_CORRECTION_LOOPS + 1):
        log(f"  Correction attempt {attempt}/{MAX_CORRECTION_LOOPS}")
        error_msg = format_errors_for_llm(build_result, file_path)

        correction_prompt = f"""Fix compilation errors in {filename}.

CURRENT CODE:
```
{current}
```

ERRORS:
{error_msg}

Return ONLY the complete corrected file. No markdown."""

        _, corrected = call_ollama(correction_prompt, label=f"Correcting {filename}")
        if not corrected:
            continue

        corrected = corrected.strip()
        if corrected.startswith("```"):
            lines = corrected.split("\n")[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            corrected = "\n".join(lines).strip()

        write_file(file_path, corrected)
        build_result = validate_file(file_path)
        if build_result.success:
            log(f"✅ Fix verified after {attempt} correction(s)")
            cleanup_backup(file_path)
            return True
        current = corrected

    log(f"❌ Could not apply fixes cleanly — restoring backup")
    restore_backup(file_path)
    return False


# ============================================================
# MAIN REVIEW ORCHESTRATOR
# ============================================================

def run_review(
    scope: str = "all",            # "java" | "angular" | "all"
    categories: list = ["all"],    # performance|security|memory|etc
    target_file: str = None,       # review single file
    auto_fix: bool = False,        # auto fix without asking
    output_file: str = None        # save report to file
) -> list:
    """
    Main entry point for code review.
    Returns list of all findings.
    """
    log(f"\n{'='*60}")
    log(f"CODE REVIEW STARTED")
    log(f"Scope     : {scope}")
    log(f"Categories: {categories}")
    log(f"{'='*60}")

    files_to_review = []

    if target_file:
        # Review single file
        if not os.path.exists(target_file):
            log(f"ERROR: File not found: {target_file}")
            return []
        files_to_review = [target_file]
    else:
        # Review entire codebase
        if scope in ["java", "all"] and os.path.exists(JAVA_REPO_ROOT):
            java_files = crawl_files(JAVA_REPO_ROOT, JAVA_EXTENSIONS)
            files_to_review.extend(java_files)
            log(f"Java files found    : {len(java_files)}")

        if scope in ["angular", "all"] and os.path.exists(ANGULAR_REPO_ROOT):
            angular_files = crawl_files(ANGULAR_REPO_ROOT, ANGULAR_EXTENSIONS)
            files_to_review.extend(angular_files)
            log(f"Angular files found : {len(angular_files)}")

    if not files_to_review:
        log("No files found to review")
        return []

    log(f"Total files to review: {len(files_to_review)}")

    # Review each file
    all_findings = []
    for i, file_path in enumerate(files_to_review, 1):
        log(f"\n[{i}/{len(files_to_review)}] {os.path.basename(file_path)}")
        findings = review_file(file_path, categories)
        all_findings.extend(findings)

    # Generate report
    if not output_file:
        output_file = os.path.join(
            os.path.dirname(LOG_FILE),
            f"review_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )

    sorted_findings = print_report(all_findings, output_file)

    # Handle fixes
    if sorted_findings:
        _handle_fixes(sorted_findings, auto_fix)

    return sorted_findings


def _handle_fixes(findings: list, auto_fix: bool):
    """Ask user which findings to fix or auto-fix if flag set."""

    # Group by file
    by_file = {}
    for f in findings:
        fp = f.get("file_path", "")
        by_file.setdefault(fp, []).append(f)

    critical_high = [
        f for f in findings
        if f.get("severity", "").upper() in ["CRITICAL", "HIGH"]
    ]

    if not critical_high:
        print("\n\033[32m  No CRITICAL or HIGH issues to fix.\033[0m")
        return

    if auto_fix:
        decision = "Y"
    else:
        print(f"\n\033[33m  Found {len(critical_high)} CRITICAL/HIGH issue(s).\033[0m")
        print("  Would you like the agent to auto-fix these?")
        print("  \033[32m[Y]\033[0m Fix all CRITICAL + HIGH issues")
        print("  \033[32m[S]\033[0m Select severity to fix")
        print("  \033[32m[N]\033[0m Skip fixing (report only)")
        decision = input("\n  Your choice (Y/S/N): ").strip().upper()

    if decision in ["N", "SKIP"]:
        print("  Fixes skipped — review report saved.")
        return

    if decision in ["S", "SELECT"]:
        print("  Fix which severity?")
        print("  [1] CRITICAL only")
        print("  [2] CRITICAL + HIGH")
        print("  [3] CRITICAL + HIGH + MEDIUM")
        sel = input("  Choice (1/2/3): ").strip()
        severities = {
            "1": ["CRITICAL"],
            "2": ["CRITICAL", "HIGH"],
            "3": ["CRITICAL", "HIGH", "MEDIUM"]
        }.get(sel, ["CRITICAL", "HIGH"])
        findings_to_fix = [
            f for f in findings
            if f.get("severity", "").upper() in severities
        ]
    else:
        findings_to_fix = critical_high

    # Group selected findings by file and fix
    fix_by_file = {}
    for f in findings_to_fix:
        fp = f.get("file_path", "")
        fix_by_file.setdefault(fp, []).append(f)

    log(f"\n⚙️  Applying fixes to {len(fix_by_file)} file(s)...")
    fix_results = []

    for file_path, file_findings in fix_by_file.items():
        log(f"\nFixing {len(file_findings)} issue(s) in {os.path.basename(file_path)}")
        success = fix_findings(file_path, file_findings)
        fix_results.append({"file": file_path, "success": success})

    # Fix summary
    log(f"\n{'='*60}")
    log("FIX SUMMARY")
    log(f"{'='*60}")
    for r in fix_results:
        icon = "✅" if r["success"] else "❌"
        log(f"{icon} {r['file']}")


# ============================================================
# ENTRY POINT (standalone use)
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Code Reviewer")
    parser.add_argument(
        "--scope",
        choices=["java", "angular", "all"],
        default="all",
        help="Which codebase to review (default: all)"
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=list(REVIEW_CATEGORIES.keys()),
        default=["all"],
        help="Review categories (default: all)"
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Review a single file instead of full codebase"
    )
    parser.add_argument(
        "--auto-fix",
        action="store_true",
        help="Automatically fix CRITICAL and HIGH issues without prompting"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save review report (default: auto-named in logs/)"
    )
    args = parser.parse_args()

    run_review(
        scope=args.scope,
        categories=args.categories,
        target_file=args.file,
        auto_fix=args.auto_fix,
        output_file=args.output
    )
