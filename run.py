#!/usr/bin/env python3
"""
Advanced Parallel NER-RE extraction pipeline via llama-server (llama.cpp).
Uses "Sharded Output" (Chunking) to eliminate NFS bottlenecks, with live-preview
logging so you can see the extractions in your console in real-time.

Usage:
    python run.py --input dataset.csv --output dataset_extracted.csv --cache cache.json --concurrency 4
"""

import argparse
import csv
import json
import logging
import os
import sys
import time
import threading
import concurrent.futures
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

# ── Delimiters & Config ───────────────────────────────────────────────────────
TUPLE_DELIMITER      = "|"
RECORD_DELIMITER     = "\n"
COMPLETION_DELIMITER = "<END>"

INPUT_COL  = "input text"
ORIGIN_COL = "origin case+#chunk"
OUTPUT_COL = "output text"

CHUNK_SIZE = 100
CHUNK_DIR = "output_chunks"

csv_lock = threading.Lock()
cache_lock = threading.Lock()
print_lock = threading.Lock()

chunk_buffer = []
chunk_counter = 1

# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """\
-Goal-
You are an expert in Named Entity and Relationship Extraction (NER-RE) with a specialization in extracting entities and relationships from legal case documents related to human smuggling. You are highly skilled at identifying and extracting only entities of the specified entity types, as well as extracting explicit relationships between them. These extracted entities and relationships will be used to build a Knowledge Graph, which will help researchers analyze human smuggling networks and identify patterns. Therefore, it is crucial to maintain strict factual accuracy and extract only what is explicitly stated in the input text, without inference or completion. You will receive entity definitions, input text, and structured examples demonstrating the correct extraction process. Study these examples carefully before performing extraction on the real input data.

Do NOT extract entities corresponding to governmental organizations or entities closely related to the trial, criminal law and law procedures, such as jury, government, law_enforcement, homeland_security, court, district court, juror, verdict, jury's verdict, hearing, proof of evidence, prosecution, supreme court, federal law, state law, public record, closing argument, greater offense, etc. We are not interested in such Government-related entities.

-Entity_type- definition
Below are the entity type definitions. Extract only entities that explicitly match them. Do NOT infer or create new entity types. If a term does not fit any defined entity type, do NOT extract it. Not all entity types will appear in every input chunk, so do NOT misclassify entities.
1. PERSON: Short name or full name of a person from any geographic regions. Smugglers, undocumented non citizens, border patrol agents, etc. are also examples of a PERSON entity.
2. LOCATION: Name of any geographical location, like cities, countries, counties, states, continents, districts, etc.
3. ORGANIZATION: Names of companies, organized criminal groups, drug cartels, smuggling rings, etc.
4. MEANS_OF_TRANSPORTATION: The mean by which someone moves from one place to another, like car, truck, 18-wheeler, etc.
5. MEANS_OF_COMMUNICATION: The mean by which communication is performed, like phone, WhatsApp, etc.
6. ROUTES: Names of roads, freeways, highways, or other types of roads.
7. SMUGGLED_ITEMS: Any illegally transported goods involved in smuggling activities. This includes drugs, weapons, and other contraband.

-Steps-
1. Extract entities only if they are explicitly written in the input document without inference or completion. For each extracted entity, extract the following information:
- entity_name: Name of the entity, capitalized. Do not alter spellings or make corrections. The name should match exactly as written.
- entity_type: One of the following types: PERSON, LOCATION, MEANS_OF_TRANSPORTATION, MEANS_OF_COMMUNICATION, ROUTES, SMUGGLED_ITEMS, ORGANIZATION
- entity_description: Comprehensive description of the entity's attributes and activities

Do not extract any entities related to government organizations or legal proceedings.
Extract each entity type separately in the following order:
- PERSON, LOCATION, MEANS_OF_TRANSPORTATION, MEANS_OF_COMMUNICATION, ROUTES, SMUGGLED_ITEMS, ORGANIZATION

Format each entity as ("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>)

2. From the entities identified in step 1, identify all pairs of (source_entity, target_entity) that are clearly related to each other.
For each pair of related entities, extract the following information:
- source_entity: name of the source entity
- target_entity: name of the target entity
- relationship_description: explanation of the relationship
- relationship_strength: A numeric score between 0 and 10 (0-3 weak, 4-6 moderate, 7-10 strong)

Format each relationship as ("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>)

3. Remove any government-related entities or relationships if mistakenly extracted.

4. Return output in English as a single list of all entities and relationships. Use **{record_delimiter}** as the list delimiter.

5. When finished, output {completion_delimiter}

######################
-Examples-
######################
Example 01:
Entity_types: PERSON, MEANS_OF_TRANSPORTATION
Input_text:
On March 12, 2024, Sai Deshpande, a known smuggler, drove an 18-wheeler carrying undocumented migrants.
######################
Output:
("entity"{tuple_delimiter}SAI DESHPANDE{tuple_delimiter}PERSON{tuple_delimiter}A known smuggler responsible for transporting migrants in an 18-wheeler)
{record_delimiter}
("entity"{tuple_delimiter}18-WHEELER{tuple_delimiter}MEANS_OF_TRANSPORTATION{tuple_delimiter}A large truck used for smuggling operations)
{record_delimiter}
("relationship"{tuple_delimiter}SAI DESHPANDE{tuple_delimiter}18-WHEELER{tuple_delimiter}Sai Deshpande drove the 18-wheeler carrying undocumented migrants{tuple_delimiter}9)
{record_delimiter}
{completion_delimiter}
"""

def build_system_prompt() -> str:
    return SYSTEM_PROMPT_TEMPLATE.format(
        tuple_delimiter=TUPLE_DELIMITER,
        record_delimiter=RECORD_DELIMITER,
        completion_delimiter=COMPLETION_DELIMITER,
    )

def setup_logging() -> logging.Logger:
    log = logging.getLogger("ner_extract")
    log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(ch)
    return log

def load_cache(cache_path: str) -> dict:
    if Path(cache_path).exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_cache(cache_path: str, cache: dict) -> None:
    tmp = cache_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp, cache_path)

def save_chunk(rows: list[dict], fieldnames: list[str]) -> None:
    """Writes a small subset of rows to the chunks directory."""
    global chunk_counter
    os.makedirs(CHUNK_DIR, exist_ok=True)
    chunk_path = os.path.join(CHUNK_DIR, f"chunk_{chunk_counter:04d}.csv")
    with open(chunk_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(rows)
    chunk_counter += 1

def compile_master_csv(master_path: str, active_rows: list[dict], fieldnames: list[str], cache: dict) -> None:
    """Updates the master list with all cached answers and saves the final file."""
    for r in active_rows:
        key = r.get(ORIGIN_COL, "").strip()
        if key in cache:
            r[OUTPUT_COL] = cache[key]
    
    tmp = master_path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        writer.writerows(active_rows)
    os.replace(tmp, master_path)

def format_eta(seconds: float) -> str:
    if seconds < 0: return "00:00:00"
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def extract_ner(client: OpenAI, model: str, system_prompt: str, input_text: str, max_retries: int, retry_delay: float, log: logging.Logger) -> tuple[str | None, int, int]:
    for attempt in range(1, max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": input_text},
                ],
                temperature=0.0,
                max_tokens=256,  # Enforce 256 token cap to prevent slow infinite loops
                stop=[
                    "<｜end▁of▁sentence｜>", 
                    "<｜end of sentence｜>",  # Space-decoded EOS
                    "<｜EOT｜>",             # DeepSeek End of Text
                    "<END>", 
                    "Example 0",    
                    "Input_text",   
                    "Output:",       
                    "\n\n\n",       
                    "###"           
                ]
            )
            content = (response.choices[0].message.content or "").strip()
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return content, prompt_tokens, completion_tokens
        except Exception as e:
            wait = retry_delay * (2 ** (attempt - 1))
            if attempt < max_retries:
                log.warning(f"Attempt {attempt}/{max_retries} failed: {e} — retrying in {wait:.1f}s")
                time.sleep(wait)
            else:
                log.error(f"All {max_retries} attempts failed: {e}")
                return None, 0, 0

def main() -> None:
    global chunk_buffer
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",       default="dataset/dataset.csv",  help="Input CSV path")
    parser.add_argument("--output",      default="dataset/dataset_extracted.csv", help="Final Master CSV path")
    parser.add_argument("--cache",       default="dataset/cache.json",  help="JSON cache file")
    parser.add_argument("--url",         default="http://localhost:8000/v1", help="API URL")
    parser.add_argument("--model",       default="deepseek-ai/DeepSeek-V4-Flash", help="Model name")
    parser.add_argument("--max-retries", type=int,   default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--concurrency", type=int,   default=4)
    args = parser.parse_args()

    log = setup_logging()
    
    print("\n" + "=" * 70)
    print("    🚀 LIVE-PREVIEW CHUNKED NER-RE EXTRACTION PIPELINE 🚀")
    print("=" * 70)
    print(f"  Input File   : {args.input}")
    print(f"  Output File  : {args.output} (Compiled at end)")
    print(f"  Chunk Folder : {CHUNK_DIR}/ (Saves every {CHUNK_SIZE} rows)")
    print(f"  Concurrency  : {args.concurrency} worker threads")
    print("=" * 70 + "\n")

    os.makedirs(CHUNK_DIR, exist_ok=True)
    cache = load_cache(args.cache)
    
    # Load Data
    with open(args.input, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames if reader.fieldnames else []
        if OUTPUT_COL not in fieldnames:
            fieldnames.append(OUTPUT_COL)
        active_rows = list(reader)

    # Identify Pending Work
    pending_items = []
    for r in active_rows:
        key = r.get(ORIGIN_COL, "").strip()
        if key not in cache:
            pending_items.append(r)

    total_rows = len(active_rows)
    completed_rows = total_rows - len(pending_items)

    log.info(f"Status Summary:")
    log.info(f"  - Total Database Rows : {total_rows}")
    log.info(f"  - Already Cached      : {completed_rows}")
    log.info(f"  - Pending             : {len(pending_items)}")

    if not pending_items:
        log.info("All rows are fully extracted. Compiling final CSV...")
        compile_master_csv(args.output, active_rows, fieldnames, cache)
        log.info("Done!")
        return

    client = OpenAI(base_url=args.url, api_key="none")
    system_prompt = build_system_prompt()

    log.info(f"Starting execution pool using {args.concurrency} parallel workers...\n")
    
    success = fail = 0
    rolling_latencies = []
    start_time = time.time()
    task_counter = 0

    def worker_task(row):
        nonlocal success, fail, task_counter
        key = row.get(ORIGIN_COL, "").strip()
        text = row.get(INPUT_COL, "").strip()

        if not text:
            with cache_lock:
                cache[key] = ""
                save_cache(args.cache, cache)
            return

        row_start = time.time()
        
        # Query the Server
        result, prompt_tokens, completion_tokens = extract_ner(
            client, args.model, system_prompt, text,
            args.max_retries, args.retry_delay, log
        )
        latency = time.time() - row_start

        if result is not None:
            row[OUTPUT_COL] = result
            with cache_lock:
                cache[key] = result
                success += 1
            status_str = "EXTRACTED"
        else:
            row[OUTPUT_COL] = "EXTRACTION_FAILED"
            with cache_lock:
                fail += 1
            status_str = "FAILED"

        # Safe Chunk Writing
        with csv_lock:
            chunk_buffer.append(row)
            if len(chunk_buffer) >= CHUNK_SIZE:
                save_chunk(chunk_buffer, fieldnames)
                chunk_buffer.clear()

        # Reporting calculation
        with cache_lock:
            task_counter += 1
            rolling_latencies.append(latency)
            if len(rolling_latencies) > 20:
                rolling_latencies.pop(0)
            
            avg_latency = sum(rolling_latencies) / len(rolling_latencies)
            eta_seconds = (avg_latency / args.concurrency) * (len(pending_items) - task_counter)
            eta_str = format_eta(eta_seconds)
            
            if task_counter % 25 == 0:
                save_cache(args.cache, cache)

        tok_per_sec = (completion_tokens / latency) if latency > 0 else 0.0

        with print_lock:
            log.info(
                f"[Row {completed_rows + task_counter}/{total_rows}] "
                f"Status: {status_str} | "
                f"Time: {latency:.2f}s | "
                f"Tokens Generated: {completion_tokens} ({tok_per_sec:.1f} tok/s) | "
                f"Success: {success} | "
                f"ETA: {eta_str}"
            )
            # LIVE PREVIEW LOG: Print a clean single-line preview of what was extracted!
            if result and status_str == "EXTRACTED":
                clean_result = result.replace('\n', ' \\ ')
                log.info(f"   ↳ Live Preview: {clean_result[:130]}...")

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = [executor.submit(worker_task, row) for row in pending_items]
            concurrent.futures.wait(futures)

    except KeyboardInterrupt:
        log.warning("\nInterrupted by user. Saving remaining buffer and compiling progress...")

    finally:
        if chunk_buffer:
            save_chunk(chunk_buffer, fieldnames)
        
        log.info("Compiling all results into the Master CSV...")
        save_cache(args.cache, cache)
        compile_master_csv(args.output, active_rows, fieldnames, cache)

    elapsed_time = time.time() - start_time
    print("\n" + "=" * 70)
    print("                     🎉 RUN COMPLETE 🎉")
    print("=" * 70)
    print(f"  Total Duration   : {format_eta(elapsed_time)}")
    print(f"  Rows Extracted   : {success}")
    print(f"  Rows Failed      : {fail}")
    print(f"  Master File      : {args.output}")
    print(f"  Chunks saved to  : {CHUNK_DIR}/")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
