import axios from "axios";
import {
  APIGatewayProxyEvent,
  APIGatewayProxyResult,
  Context,
} from "aws-lambda";

type ImageSearchRequestBody = {
  base64?: string;
};

type ImageEmbedResponse = {
  vectors?: number[][];
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
  error?: string;
};

type HitLike = { _source?: Record<string, unknown>; _score?: number };

const OUTAGE_ERROR = process.env.OUTAGE_ERROR ?? "";
/** Base URL of in-service GPU worker (e.g. http://search-worker-alb-xxx.elb.amazonaws.com) — POST /search_b64 */
const GPU_WORKER_URL = (process.env.GPU_WORKER_URL ?? "").replace(/\/$/, "");
const IMAGE_EMBED_ENDPOINT =
  process.env.IMAGE_EMBED_ENDPOINT ?? "";
const OS_IMAGE_SEARCH_URL =
  process.env.OS_IMAGE_SEARCH_URL ??
  "";
const OS_IMAGE_SEARCH_AUTH =
  process.env.OS_IMAGE_SEARCH_AUTH ??
  "";

const MATCH_SCORE_THRESHOLD = 0.85;
const TITLE_MATCH_MIN_FRACTION = 0.5;

/** Same stop list as tools/search_ui.html — buildTitleDensity */
const TITLE_DENSITY_STOP = new Set([
  "the", "and", "for", "with", "from", "this", "that", "are", "was",
  "have", "has", "been", "will", "can", "not", "all", "but", "its",
  "per", "via", "lot", "set", "new", "one", "two", "buy", "now",
]);

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

const fetchImageVector = async (base64: string): Promise<number[]> => {
  const payload = { image_urls: ["data:image/jpeg;base64," + base64] };
  // console.log(payload)
  const res = await axios.post<ImageEmbedResponse>(IMAGE_EMBED_ENDPOINT, payload, {
    headers: { "Content-Type": "application/json" },
  });
  // console.log("Embed response data:", res.data);
  const vec = res.data?.vectors?.[0];
  if (!Array.isArray(vec)) {
    throw new Error("Embed endpoint did not return a vector");
  }
  // return vec.map((v) => Number(v)).filter((v) => Number.isFinite(v));
  return vec;
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
      headers: { "Content-Type": "application/json" },
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

/** Title density "compiled" line: words in ≥50% of high-confidence titles (mirrors search_ui.html buildTitleDensity). */
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
  const highConf = results.filter(
    (r) => Number(r.score ?? 0) >= MATCH_SCORE_THRESHOLD,
  );
  const hcN = highConf.length;
  if (hcN === 0) return "";

  const hcWordCounts: Record<string, number> = {};
  for (const r of highConf) {
    const title = String((r.doc?.title as string | undefined) ?? "").toLowerCase();
    if (!title) continue;
    const words = tokenizeTitleWords(title);
    for (const w of words) {
      hcWordCounts[w] = (hcWordCounts[w] ?? 0) + 1;
    }
  }

  return Object.entries(hcWordCounts)
    .filter(([, c]) => c / hcN >= TITLE_MATCH_MIN_FRACTION)
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

type SearchResults = { hits?: { hits?: any[] } };

const searchImageVector = async (vector: number[]): Promise<SearchResults> => {
  const authHeader = OS_IMAGE_SEARCH_AUTH.startsWith("Basic")
    ? OS_IMAGE_SEARCH_AUTH
    : `Basic ${OS_IMAGE_SEARCH_AUTH}`;

  const payload = {
    size: 20,
    min_score: 0,
    track_scores: true,
    query: {
      knn: {
        imageVector: {
          vector,
          k: 20,
        },
      },
    },
    sort: [{ _score: { order: "desc" } }],
  };
  

  const res = await axios.post(OS_IMAGE_SEARCH_URL, payload, {
    headers: {
      "Content-Type": "application/json",
      Authorization: authHeader,
    },
  });

  return res.data as SearchResults;
};

const getEndTime = (hit: HitLike): number => {
  const raw = hit?._source?.endTime ?? hit?._source?.EndTime;
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    // Date.parse does not reliably parse "YYYY-MM-DD HH:mm:ss" (space) across runtimes.
    // Replace space with "T" for ISO 8601 compatibility.
    const isoLike = raw.replace(/^(\d{4}-\d{2}-\d{2})\s+/, "$1T");
    const t = Date.parse(isoLike);
    return Number.isFinite(t) ? t : 0;
  }
  return 0;
};

const getSalePrice = (hit: HitLike): number => {
  const raw = hit?._source?.salePrice ?? hit?._source?.SalePrice ?? hit?._source?.currentPrice ?? hit?._source?.CurrentPrice;
  if (typeof raw === "number") return raw;
  if (typeof raw === "string") {
    const n = Number.parseFloat(raw);
    return Number.isFinite(n) ? n : 0;
  }
  return 0;
};

/** Sort comparator: most recent date first, oldest last (descending by endTime). */
const compareEndTimeDesc = (a: HitLike, b: HitLike): number =>
  getEndTime(b) - getEndTime(a);

const sortResultsByEndTime = (results: any): any => {
  const hits: HitLike[] | unknown = results?.hits?.hits;
  if (Array.isArray(hits)) {
    hits.sort((a, b) => compareEndTimeDesc(a as HitLike, b as HitLike));
  }
  return results;
};

type CardIdentEntry = { field: string; value: string; hits: number };

const normalizeValue = (val: unknown): string | null => {
  if (Array.isArray(val)) {
    const first = val.find((v) => typeof v === "string" && v.trim().length > 0) as string | undefined;
    return first ? first.trim() : null;
  }
  if (typeof val === "string" || typeof val === "number" || typeof val === "boolean") {
    const s = String(val).trim();
    return s.length > 0 ? s : null;
  }
  return null;
};

const buildCardIdent = (results: any): CardIdentEntry[] => {
  const hits: any[] = results?.hits?.hits ?? [];
  const cardIdentMap: Record<string, Record<string, number>> = {};

  hits.forEach((hit) => {
    const specifics = hit?._source?.itemSpecifics;
    if (Array.isArray(specifics)) {
      specifics.forEach((spec: Record<string, unknown>) => {
        const name = normalizeValue((spec as { name?: unknown; Name?: unknown }).name ?? (spec as { Name?: unknown }).Name);
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

  const cardIdent: CardIdentEntry[] = Object.entries(cardIdentMap).map(([field, counts]) => {
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

  return cardIdent;
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
    const gradedChunk = grade ? `${label} ${grade}` : label;
    parts.push(gradedChunk);
  }

  // Remove duplicate words (case-insensitive) while preserving order
  const joined = parts.join(" ");
  const words = joined.split(/\s+/);
  const seen = new Set<string>();
  const uniqueWords = words.filter((word) => {
    const lower = word.toLowerCase();
    if (seen.has(lower)) return false;
    seen.add(lower);
    return true;
  });

  return uniqueWords.join(" ");
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

  const body = parseBody(event.body ?? null);
  if (!body || !body.base64) {
    return {
      statusCode: 400,
      body: JSON.stringify({ message: "Invalid or missing request body/base64" }),
    };
  }

  const useGpuWorker = Boolean(GPU_WORKER_URL);
  const useLegacyOs =
    Boolean(IMAGE_EMBED_ENDPOINT) &&
    Boolean(OS_IMAGE_SEARCH_URL) &&
    Boolean(OS_IMAGE_SEARCH_AUTH);

  if (!useGpuWorker && !useLegacyOs) {
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Image search failed",
        error:
          "Set GPU_WORKER_URL for in-service GPU search, or set IMAGE_EMBED_ENDPOINT, OS_IMAGE_SEARCH_URL, and OS_IMAGE_SEARCH_AUTH for legacy OpenSearch KNN",
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    };
  }

  const base64Clean = cleanBase64(body.base64);
  const topK = 20;
  const hnswEfEnv = process.env.GPU_HNSW_EF;
  const hnswEfParsed = hnswEfEnv ? Number.parseInt(hnswEfEnv, 10) : undefined;
  const hnswEf =
    hnswEfParsed !== undefined && Number.isFinite(hnswEfParsed)
      ? hnswEfParsed
      : undefined;

  try {
    let items: any[] = [];
    let scores: { id: unknown; score: unknown }[] = [];
    let cardIdent: CardIdentEntry[] = [];
    let processedTitle = "";
    let titleMatch = "";
    let options: Record<string, unknown>;

    if (useGpuWorker) {
      const startGpu = Date.now();
      const gpuData = await fetchGpuWorkerSearch(base64Clean, topK, hnswEf);
      timings.gpuWorkerRoundTrip = Date.now() - startGpu;
      if (gpuData.encode_ms != null) timings.gpu_encode_ms = gpuData.encode_ms;
      if (gpuData.qdrant_ms != null) timings.gpu_qdrant_ms = gpuData.qdrant_ms;
      if (gpuData.enrich_ms != null) timings.gpu_enrich_ms = gpuData.enrich_ms;
      console.log(`[TIMING] gpuWorkerRoundTrip: ${timings.gpuWorkerRoundTrip}ms`);

      if (gpuData.error) {
        throw new Error(String(gpuData.error));
      }

      const qdrantResults = gpuData.qdrant_results ?? [];
      const allHits = qdrantResultsToHits(qdrantResults);
      let filteredHits = allHits.filter((hit) => getSalePrice(hit) > 0);
      filteredHits = [...filteredHits]
        .sort((a, b) => compareEndTimeDesc(a, b))
        .slice(0, topK);

      const startProcessHits = Date.now();
      filteredHits.forEach((result) => {
        const rec = result as Record<string, unknown>;
        rec["score"] = rec._score;
        scores.push({
          id: (result as { _source?: { id?: unknown } })._source?.id,
          score: rec._score,
        });
        items.push(result);
      });
      timings.processHits = Date.now() - startProcessHits;

      const highConfQr = qdrantResults.filter(
        (r) => Number(r.score ?? 0) >= MATCH_SCORE_THRESHOLD,
      );
      const highConfHits = qdrantResultsToHits(highConfQr);
      const resultsForCardIdent: SearchResults = {
        hits: { hits: highConfHits as any[] },
      };

      const startCardIdent = Date.now();
      cardIdent = buildCardIdent(resultsForCardIdent);
      timings.buildCardIdent = Date.now() - startCardIdent;

      const startTitle = Date.now();
      processedTitle = buildProcessedTitle(cardIdent, highConfQr.length);
      titleMatch = buildTitleMatch(qdrantResults);
      timings.buildProcessedTitle = Date.now() - startTitle;

      options = {
        gpuWorkerUrl: GPU_WORKER_URL,
        top_k: topK,
        ...(hnswEf != null ? { hnsw_ef: hnswEf } : {}),
        image_size: gpuData.image_size,
      };
    } else {
      const startEmbed = Date.now();
      const vector = await fetchImageVector(base64Clean);
      timings.fetchImageVector = Date.now() - startEmbed;
      console.log(`[TIMING] fetchImageVector: ${timings.fetchImageVector}ms`);

      const startSearch = Date.now();
      const unsortedResults = await searchImageVector(vector);
      timings.searchImageVector = Date.now() - startSearch;
      console.log(`[TIMING] searchImageVector: ${timings.searchImageVector}ms`);

      const startSort = Date.now();
      const sortedResults = sortResultsByEndTime(unsortedResults);
      timings.sortResultsByEndTime = Date.now() - startSort;
      console.log(`[TIMING] sortResultsByEndTime: ${timings.sortResultsByEndTime}ms`);

      const sortedData = sortedResults as SearchResults;
      const allHits = sortedData?.hits?.hits ?? [];
      const knnHits = Array.isArray(allHits) ? allHits : [];

      const highConfHits = knnHits.filter(
        (hit) => Number((hit as HitLike)._score ?? 0) >= MATCH_SCORE_THRESHOLD,
      );
      const resultsHighConf: SearchResults = {
        hits: { hits: highConfHits as any[] },
      };

      const startCardIdent = Date.now();
      cardIdent = buildCardIdent(resultsHighConf);
      timings.buildCardIdent = Date.now() - startCardIdent;

      let filteredHits = knnHits.filter((hit) => getSalePrice(hit as HitLike) > 0);
      filteredHits = [...filteredHits]
        .sort((a, b) => compareEndTimeDesc(a as HitLike, b as HitLike))
        .slice(0, topK);
      const results = {
        ...sortedResults,
        hits: { ...sortedData?.hits, hits: filteredHits },
      };

      const startProcessHits = Date.now();
      const resultHits = (results as SearchResults)?.hits?.hits;
      if (Array.isArray(resultHits)) {
        resultHits.forEach((result) => {
          (result as Record<string, unknown>)["score"] = (
            result as Record<string, unknown>
          )._score;
          scores.push({
            id: (result as { _source?: { id?: unknown } })._source?.id,
            score: (result as Record<string, unknown>)._score,
          });
          items.push(result);
        });
      }
      timings.processHits = Date.now() - startProcessHits;

      const startTitle = Date.now();
      processedTitle = buildProcessedTitle(cardIdent, highConfHits.length);
      const legacyForTitleMatch: QdrantSearchResultItem[] = highConfHits.map(
        (hit) => ({
          score: Number((hit as HitLike)._score ?? 0),
          doc: ((hit as HitLike)._source ?? {}) as Record<string, unknown>,
        }),
      );
      titleMatch = buildTitleMatch(legacyForTitleMatch);
      timings.buildProcessedTitle = Date.now() - startTitle;

      options = { searchUrl: OS_IMAGE_SEARCH_URL, size: topK, k: topK };
    }

    timings.total = Date.now() - startTotal;
    console.log(`[TIMING] total: ${timings.total}ms`);
    console.log(`[TIMING] Summary:`, JSON.stringify(timings));

    return {
      statusCode: 200,
      body: JSON.stringify({
        message: "Success2",
        items,
        scores,
        title: processedTitle,
        titleMatch,
        options,
        cardIdent,
        timings,
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    };
  } catch (error) {
    timings.total = Date.now() - startTotal;
    console.error("Image vector search failed", error);
    console.log(`[TIMING] failed after: ${timings.total}ms`, JSON.stringify(timings));
    return {
      statusCode: 500,
      body: JSON.stringify({
        message: "Image search failed",
        error: error instanceof Error ? error.message : "unknown",
        timings
      }),
      headers: {
        "Access-Control-Allow-Origin": "*",
        "Content-Type": "application/json",
      },
    };
  }
};
