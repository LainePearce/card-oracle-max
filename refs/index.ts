import axios from "axios";
import {
  APIGatewayProxyEvent,
  APIGatewayProxyResult,
  Context,
} from "aws-lambda";

type ImageSearchRequestBody = {
  base64?: string;
};

type QdrantSearchResultItem = {
  qdrant_id?: string;
  os_id?: string;
  score?: number;
  payload?: Record<string, unknown>;
  doc?: Record<string, unknown>;
};

type GpuWorkerSearchResponse = {
  image_size?: string;
  qdrant_results?: QdrantSearchResultItem[];
  total?: number;
  encode_ms?: number;
  qdrant_ms?: number;
  enrich_ms?: number;
  score_threshold?: number;
  qdrant_calls?: number;
  error?: string;
};

type HitLike = { _source?: Record<string, unknown>; _score?: number };
type ResultItem = { _source?: Record<string, unknown>; _score?: number; score?: number };

const OUTAGE_ERROR = process.env.OUTAGE_ERROR ?? "";
const GPU_WORKER_URL = (process.env.GPU_WORKER_URL ?? "").trim().replace(/\/$/, "");

const parseEnvPositiveInt = (name: string, defaultVal: number): number => {
  const raw = process.env[name];
  if (raw == null || raw.trim() === "") return defaultVal;
  const n = Number.parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : defaultVal;
};

/** From `TOP_K` env, default 25 (GPU worker returns at most this many). */
const TOP_K = parseEnvPositiveInt("TOP_K", 25);
/**
 * Only the strongest matches (by score) are used to build `cardIdent`, `title`, and `titleMatch`.
 * Default matches the GPU worker's max result count (25); override if the worker limit changes.
 */
const TITLE_IDENT_TOP_N = parseEnvPositiveInt("TITLE_IDENT_TOP_N", 25);
const TITLE_MATCH_MIN_FRACTION = 0.5;

const TITLE_DENSITY_STOP = new Set([
  "the", "and", "for", "with", "from", "this", "that", "are", "was",
  "have", "has", "been", "will", "can", "not", "all", "but", "its",
  "per", "via", "lot", "set", "new", "one", "two", "buy", "now",
]);

/**
 * When UTF-8 bytes were decoded as Latin-1/ISO-8859-1, multibyte chars become pairs
 * (e.g. é → U+00C3 + U+00A9, shown as "Ã©" or often described as Ã + ©). Reinterpret
 * code units as bytes and decode as UTF-8.
 */
const repairUtf8Mojibake = (s: string): string => {
  if (!/[\u00c2\u00c3][\u0080-\u00bf]/.test(s)) return s;
  const bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) {
    bytes[i] = s.charCodeAt(i) & 0xff;
  }
  return new TextDecoder("utf-8", { fatal: false }).decode(bytes);
};

const deepRepairUtf8Mojibake = <T>(value: T): T => {
  if (typeof value === "string") {
    return repairUtf8Mojibake(value) as T;
  }
  if (Array.isArray(value)) {
    return value.map((x) => deepRepairUtf8Mojibake(x)) as T;
  }
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(o)) {
      out[k] = deepRepairUtf8Mojibake(o[k]);
    }
    return out as T;
  }
  return value;
};

const cleanBase64 = (input: string): string => {
  const trimmed = input.trim();
  const noPrefix = trimmed.includes(",") ? trimmed.split(",").pop() ?? "" : trimmed;
  return noPrefix.replace(/\s+/g, "");
};

const parseBody = (raw: string | null): ImageSearchRequestBody | null => {
  if (!raw) return null;
  try {
    return JSON.parse(raw) as ImageSearchRequestBody;
  } catch (error) {
    console.error("Failed to parse request body", error);
    return null;
  }
};

const fetchGpuWorkerSearch = async (
  imageB64: string,
  topK: number,
  hnswEf?: number,
): Promise<GpuWorkerSearchResponse> => {
  const body: Record<string, unknown> = { image_b64: imageB64, top_k: topK };
  if (hnswEf != null) body.hnsw_ef = hnswEf;
  const res = await axios.post<GpuWorkerSearchResponse>(
    `${GPU_WORKER_URL}/search_b64`,
    body,
    {
      headers: { "Content-Type": "application/json; charset=utf-8" },
      timeout: 120_000,
      validateStatus: () => true,
    },
  );
  if (res.status >= 400) {
    const msg =
      (res.data && typeof res.data === "object" && "error" in res.data
        ? String((res.data as GpuWorkerSearchResponse).error)
        : null) ?? `GPU worker HTTP ${res.status}`;
    throw new Error(msg);
  }
  return res.data;
};

const tokenizeTitleWords = (titleLower: string): Set<string> => {
  const words = titleLower
    .split(/[^a-z0-9]+/)
    .filter(
      (w) =>
        w.length >= 3 &&
        !TITLE_DENSITY_STOP.has(w) &&
        !/^\d{1,2}$/.test(w),
    );
  return new Set(words);
};

const buildTitleMatch = (results: QdrantSearchResultItem[]): string => {
  const n = results.length;
  if (n === 0) return "";

  const hcWordCounts: Record<string, number> = {};
  for (const r of results) {
    const title = String((r.doc?.title as string | undefined) ?? "").toLowerCase();
    if (!title) continue;
    const words = tokenizeTitleWords(title);
    for (const w of words) {
      hcWordCounts[w] = (hcWordCounts[w] ?? 0) + 1;
    }
  }

  return Object.entries(hcWordCounts)
    .filter(([, c]) => c / n >= TITLE_MATCH_MIN_FRACTION)
    .sort((a, b) => b[1] - a[1])
    .map(([w]) => w)
    .join(" ");
};

const qdrantResultsToHits = (
  items: QdrantSearchResultItem[],
): HitLike[] =>
  items.map((r) => ({
    _source: (r.doc ?? {}) as Record<string, unknown>,
    _score: r.score,
  }));

const getEndTime = (hit: HitLike): number => {
  const raw = hit?._source?.endTime ?? hit?._source?.EndTime;
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const isoLike = raw.replace(/^(\d{4}-\d{2}-\d{2})\s+/, "$1T");
    const t = Date.parse(isoLike);
    return Number.isFinite(t) ? t : 0;
  }
  return 0;
};

const getSalePrice = (hit: HitLike): number => {
  const raw =
    hit?._source?.salePrice ??
    hit?._source?.SalePrice ??
    hit?._source?.currentPrice ??
    hit?._source?.CurrentPrice;
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
};

/** Sort comparator: most recent date first (descending by endTime). */
const compareEndTimeDesc = (a: HitLike, b: HitLike): number =>
  getEndTime(b) - getEndTime(a);

type CardIdentEntry = { field: string; value: string; hits: number };

const normalizeValue = (val: unknown): string | null => {
  if (Array.isArray(val)) {
    const first = val.find((v) => typeof v === "string" && v.trim().length > 0) as string | undefined;
    return first ? first.trim() : null;
  }
  if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
    const s = String(val).trim();
    if (s.length === 0) return null;
    const lower = s.toLowerCase();
    if (lower === "undefined" || lower === "null") return null;
    return s;
  }
  return null;
};

const isBadDisplayString = (s: string): boolean => {
  const t = s.trim();
  if (t.length === 0) return true;
  const lower = t.toLowerCase();
  return lower === "undefined" || lower === "null";
};

/** Strip undefined, invalid numbers, and literal "undefined"/"null" strings (deep). */
const stripUndefinedDeep = (value: unknown): unknown => {
  if (value === undefined) return undefined;
  if (typeof value === "string") {
    return isBadDisplayString(value) ? undefined : value;
  }
  if (typeof value === "number") {
    return Number.isFinite(value) ? value : undefined;
  }
  if (Array.isArray(value)) {
    return value
      .map((x) => stripUndefinedDeep(x))
      .filter((x) => x !== undefined);
  }
  if (value !== null && typeof value === "object") {
    const o = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const k of Object.keys(o)) {
      const inner = stripUndefinedDeep(o[k]);
      if (inner === undefined) continue;
      out[k] = inner;
    }
    return out;
  }
  return value;
};

const sanitizeItemSpecificsArray = (arr: unknown[]): unknown[] => {
  return arr
    .map((row) => stripUndefinedDeep(row))
    .filter((row) => {
      if (!row || typeof row !== "object") return false;
      const spec = row as Record<string, unknown>;
      const name = normalizeValue(spec.name ?? spec.Name);
      const value = normalizeValue(
        spec.value ?? spec.Value ?? spec.values ?? spec.Values,
      );
      return name != null && value != null;
    });
};

const hasRenderablePrice = (s: Record<string, unknown>): boolean => {
  const keys = [
    "salePrice", "SalePrice", "currentPrice", "CurrentPrice", "price", "Price",
  ] as const;
  for (const k of keys) {
    const v = s[k];
    if (v === undefined || v === null) continue;
    if (typeof v === "number" && Number.isFinite(v)) return true;
    if (typeof v === "string" && !isBadDisplayString(v)) return true;
  }
  return false;
};

const sanitizeHitSource = (source: Record<string, unknown>): Record<string, unknown> => {
  const out = stripUndefinedDeep(source) as Record<string, unknown>;
  for (const key of ["itemSpecifics", "ItemSpecifics"] as const) {
    const specs = out[key];
    if (Array.isArray(specs)) {
      out[key] = sanitizeItemSpecificsArray(specs);
    }
  }
  if (!hasRenderablePrice(out)) {
    for (const k of [
      "currency", "Currency", "priceCurrency", "PriceCurrency",
      "saleCurrency", "SaleCurrency",
    ]) {
      delete out[k];
    }
  }
  return out;
};

const sourceHasRenderableContent = (s: Record<string, unknown>): boolean => {
  for (const v of Object.values(s)) {
    if (v === undefined || v === null) continue;
    if (typeof v === "string" && !isBadDisplayString(v)) return true;
    if (typeof v === "number" && Number.isFinite(v)) return true;
    if (typeof v === "boolean") return true;
    if (Array.isArray(v) && v.length > 0) return true;
    if (typeof v === "object" && !Array.isArray(v) && v !== null) {
      if (Object.keys(v as object).length > 0) return true;
    }
  }
  return false;
};

const buildCardIdent = (hits: HitLike[]): CardIdentEntry[] => {
  const cardIdentMap: Record<string, Record<string, number>> = {};

  hits.forEach((hit) => {
    const specifics = hit?._source?.itemSpecifics;
    if (Array.isArray(specifics)) {
      specifics.forEach((spec: Record<string, unknown>) => {
        const name = normalizeValue(
          (spec as { name?: unknown; Name?: unknown }).name ??
          (spec as { Name?: unknown }).Name,
        );
        const value = normalizeValue(
          (spec as { value?: unknown; Value?: unknown }).value ??
          (spec as { Value?: unknown }).Value ??
          (spec as { values?: unknown; Values?: unknown }).values ??
          (spec as { Values?: unknown }).Values,
        );
        if (name && value) {
          cardIdentMap[name] = cardIdentMap[name] || {};
          cardIdentMap[name][value] = (cardIdentMap[name][value] ?? 0) + 1;
        }
      });
    } else if (specifics && typeof specifics === "object") {
      Object.entries(specifics as Record<string, unknown>).forEach(([key, val]) => {
        const name = normalizeValue(key);
        const value = normalizeValue(val);
        if (name && value) {
          cardIdentMap[name] = cardIdentMap[name] || {};
          cardIdentMap[name][value] = (cardIdentMap[name][value] ?? 0) + 1;
        }
      });
    }
  });

  return Object.entries(cardIdentMap).map(([field, counts]) => {
    let bestValue = "";
    let bestCount = -1;
    Object.entries(counts).forEach(([val, count]) => {
      if (count > bestCount) {
        bestCount = count;
        bestValue = val;
      }
    });
    return { field, value: bestValue, hits: bestCount };
  });
};

const buildProcessedTitle = (cardIdent: CardIdentEntry[], totalHits: number): string => {
  if (!totalHits || totalHits <= 0) return "";
  const threshold = totalHits * 0.2;
  const blockedFields = new Set(["country", "graded", "type", "autographed", "team", "genre"]);

  const isBoolish = (val: string): boolean => {
    const l = val.toLowerCase();
    return l === "true" || l === "false";
  };

  const extractAcronym = (val: string): string => {
    const paren = val.match(/\(([^)]+)\)\s*$/);
    if (paren && paren[1]) return paren[1].trim();
    const tokens = val.trim().split(/\s+/);
    const last = tokens[tokens.length - 1];
    if (last && last.length <= 5 && last === last.toUpperCase()) return last;
    return val.trim();
  };

  const parts: string[] = [];
  let graded = false;
  let grader: string | undefined;
  let grade: string | undefined;

  cardIdent.forEach((entry) => {
    if (!entry.value || entry.hits < threshold) return;
    const field = entry.field?.toLowerCase?.() ?? "";
    const value = entry.value.trim();
    if (!value) return;

    if (field === "graded") {
      graded = value.toLowerCase() === "true";
      return;
    }
    if (field.includes("grader") || field.includes("grading")) {
      grader = value;
      return;
    }
    if (field === "grade") {
      if (!isBoolish(value)) grade = value;
      return;
    }
    if (blockedFields.has(field)) return;
    if (isBoolish(value)) return;

    parts.push(value);
  });

  if (graded && grader) {
    const label = extractAcronym(grader);
    parts.push(grade ? `${label} ${grade}` : label);
  }

  const joined = parts.join(" ");
  const words = joined.split(/\s+/);
  const seen = new Set<string>();
  const uniqueWords = words.filter((word) => {
    const lower = word.toLowerCase();
    if (seen.has(lower)) return false;
    seen.add(lower);
    return true;
  });

  return finalizeResponseTitle(uniqueWords.join(" "));
};

/** Strips brackets, normalizes Pokémon publisher strings, and removes duplicate tokens. */
const finalizeResponseTitle = (raw: string): string => {
  let s = repairUtf8Mojibake(raw);
  s = s.replace(/[\[\]]/g, "");
  s = s.replace(/\bthe\s+pok[eé]mon\s+company\s+international\b/gi, "Pokemon");
  s = s.replace(/\bpok[eé]mon\s+company\s+international\b/gi, "Pokemon");
  s = s.replace(/\bthe\s+pok[eé]mon\s+company\b/gi, "Pokemon");
  s = s.replace(/\bpok[eé]mon\s+company\b/gi, "Pokemon");
  const words = s.split(/\s+/).filter(Boolean);
  const seen = new Set<string>();
  const unique = words.filter((word) => {
    const key = word.toLowerCase();
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
  return unique.join(" ");
};

export const handler = async (
  event: APIGatewayProxyEvent,
  _context: Context,
): Promise<APIGatewayProxyResult> => {
  const timings: Record<string, number> = {};
  const startTotal = Date.now();

  if (OUTAGE_ERROR && OUTAGE_ERROR !== "") {
    return {
      statusCode: 503,
      body: JSON.stringify({ message: OUTAGE_ERROR }),
    };
  }

  if (!GPU_WORKER_URL) {
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Image search failed",
        error: "GPU_WORKER_URL is not configured",
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json; charset=utf-8",
      },
    };
  }

  const body = parseBody(event.body ?? null);
  if (!body || !body.base64) {
    return {
      statusCode: 400,
      body: JSON.stringify({ message: "Invalid or missing request body/base64" }),
    };
  }

  const base64Clean = cleanBase64(body.base64);
  const hnswEfEnv = process.env.GPU_HNSW_EF;
  const hnswEfParsed = hnswEfEnv ? Number.parseInt(hnswEfEnv, 10) : undefined;
  const hnswEf =
    hnswEfParsed !== undefined && Number.isFinite(hnswEfParsed)
      ? hnswEfParsed
      : undefined;

  try {
    const startGpu = Date.now();
    const gpuData = await fetchGpuWorkerSearch(base64Clean, TOP_K, hnswEf);
    timings.gpuWorkerRoundTrip = Date.now() - startGpu;
    if (gpuData.encode_ms      != null) timings.gpu_encode_ms      = gpuData.encode_ms;
    if (gpuData.qdrant_ms      != null) timings.gpu_qdrant_ms      = gpuData.qdrant_ms;
    if (gpuData.enrich_ms      != null) timings.gpu_enrich_ms      = gpuData.enrich_ms;
    if (gpuData.score_threshold != null) timings.gpu_score_threshold = gpuData.score_threshold;
    if (gpuData.qdrant_calls   != null) timings.gpu_qdrant_calls   = gpuData.qdrant_calls;
    console.log(`[TIMING] gpuWorkerRoundTrip: ${timings.gpuWorkerRoundTrip}ms`);

    if (gpuData.error) {
      throw new Error(String(gpuData.error));
    }

    // Keep every result Qdrant returned. Only drop if we have literally no data
    // from either the OS-enriched doc or the Qdrant payload — this handles the
    // rare case of a malformed response entry. The sourceHasRenderableContent
    // check is intentionally omitted: sparse payloads (e.g. fallback results
    // where OS enrichment failed) are still valid hits and must not be dropped.
    const qdrantResults: QdrantSearchResultItem[] = (gpuData.qdrant_results ?? []).flatMap(
      (r) => {
        const hasDoc     = r.doc     != null && Object.keys(r.doc).length     > 0;
        const hasPayload = r.payload != null && Object.keys(r.payload).length > 0;
        if (!hasDoc && !hasPayload) return [];

        // Use OS doc when available; fall back to Qdrant payload when OS
        // enrichment failed (e.g. transient outage or ID mismatch).
        const source = hasDoc ? r.doc! : r.payload!;
        const repaired = deepRepairUtf8Mojibake(source) as Record<string, unknown>;
        const cleaned  = sanitizeHitSource(repaired);

        return [{ ...r, doc: cleaned }];
      },
    );

    const allHits = qdrantResultsToHits(qdrantResults);
    const filteredHits = [...allHits].sort(compareEndTimeDesc);

    const startProcessHits = Date.now();
    const items: ResultItem[] = filteredHits.map((hit) => ({
      _source: hit._source,
      _score:  hit._score,
      score:   hit._score,
    }));
    const scores: { id: unknown; score: unknown }[] = filteredHits.map((hit) => ({
      id:    hit._source?.id,
      score: hit._score,
    }));
    timings.processHits = Date.now() - startProcessHits;

    // Card identity and title derived from the top-scored results
    const sortedByScore = [...qdrantResults].sort(
      (a, b) => Number(b.score ?? 0) - Number(a.score ?? 0),
    );
    const derivationQr   = sortedByScore.slice(0, TITLE_IDENT_TOP_N);
    const derivationHits = qdrantResultsToHits(derivationQr);

    const startCardIdent = Date.now();
    const cardIdent = buildCardIdent(derivationHits);
    timings.buildCardIdent = Date.now() - startCardIdent;

    const startTitle = Date.now();
    const processedTitle = buildProcessedTitle(cardIdent, derivationQr.length);
    const titleMatch     = buildTitleMatch(derivationQr);
    timings.buildProcessedTitle = Date.now() - startTitle;

    timings.total = Date.now() - startTotal;
    console.log(`[TIMING] total: ${timings.total}ms`);
    console.log(`[TIMING] Summary:`, JSON.stringify(timings));

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: "Success",
        items,
        scores,
        title: processedTitle,
        titleMatch,
        options: {
          gpuWorkerUrl:    GPU_WORKER_URL,
          top_k:           TOP_K,
          ...(hnswEf != null ? { hnsw_ef: hnswEf } : {}),
          image_size:      gpuData.image_size,
          score_threshold: gpuData.score_threshold,
          qdrant_calls:    gpuData.qdrant_calls,
        },
        cardIdent,
        timings,
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json; charset=utf-8",
      },
    };
  } catch (error) {
    timings.total = Date.now() - startTotal;
    console.error("Image search failed", error);
    console.log(`[TIMING] failed after: ${timings.total}ms`, JSON.stringify(timings));
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Image search failed",
        error: error instanceof Error ? error.message : "unknown",
        timings,
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json; charset=utf-8",
      },
    };
  }
};
