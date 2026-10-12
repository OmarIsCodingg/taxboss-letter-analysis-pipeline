/**
 * Standalone Dutch tax-letter analysis pipeline.
 *
 * Pipeline:
 * 1. Claude Sonnet extracts text from an uploaded letter image.
 * 2. Claude Haiku classifies the extracted or supplied text.
 * 3. Claude Sonnet generates a short plain-language explanation.
 *
 * This sample intentionally excludes authentication, rate limiting,
 * database persistence, and application-specific configuration.
 */

const ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION = "2023-06-01";
const OCR_MODEL = "claude-sonnet-4-5";
const CLASSIFICATION_MODEL = "claude-haiku-4-5";
const EXPLANATION_MODEL = "claude-sonnet-4-5";

type ImageMimeType =
  | "image/jpeg"
  | "image/png"
  | "image/gif"
  | "image/webp";

type Urgency = "GROEN" | "ORANJE" | "ROOD";

type LetterType =
  | "BTW_AMBTSHALVE"
  | "BTW_BEZWAAR_UITSPRAAK"
  | "BTW_TERUGGAAF"
  | "BTW_NAHEFFING"
  | "BTW_HERINNERING"
  | "BTW_SUPPLETIE"
  | "BETALINGSHERINNERING"
  | "AANMANING"
  | "DWANGBEVEL"
  | "INFORMATIEVERZOEK"
  | "BOEKENONDERZOEK"
  | "BOETE_VERZUIM"
  | "BOETE_VERGRIJP"
  | "IB_VOORLOPIG"
  | "IB_DEFINITIEF"
  | "IB_TERUGGAAF"
  | "KOR_BESCHIKKING"
  | "NIET_BELASTINGDIENST"
  | "ONBEKEND";

interface LetterClassification {
  type: LetterType;
  afzender: string;
  periode: string | null;
  dagtekening: string | null;
  heeft_deadline: boolean;
  deadline_datum: string | null;
  actie_vereist: boolean;
  urgentie: Urgency;
  zelf_ingediend: boolean;
  kor_uitkomst: "toegekend" | "geweigerd" | null;
}

interface AnalysisResult {
  extractedText: string;
  classification: LetterClassification;
  explanation: string;
}

interface ClaudeTextBlock {
  type: "text";
  text: string;
}

interface ClaudeResponse {
  content?: ClaudeTextBlock[];
  usage?: {
    input_tokens?: number;
    output_tokens?: number;
  };
}

const CLASSIFICATION_SYSTEM_PROMPT = `
Je classificeert Nederlandse belastingbrieven voor ZZP'ers.
Retourneer ALLEEN een JSON-object.
Gebruik geen markdown en schrijf geen uitleg buiten het JSON-object.

MOGELIJKE TYPES:

BTW_AMBTSHALVE
Belastingdienst herziet een eerdere BTW-beslissing zelf, bijvoorbeeld op basis
van artikel 17 Wet op de omzetbelasting of artikel 65 AWR.

BTW_BEZWAAR_UITSPRAAK
Uitspraak op een bezwaar dat de ontvanger zelf heeft ingediend.

BTW_TERUGGAAF
Gewone BTW-teruggaaf na een aangifte.

BTW_NAHEFFING
Naheffingsaanslag na een controle of correctie.

BTW_HERINNERING
Herinnering omdat een BTW-aangifte nog niet is ontvangen.

BTW_SUPPLETIE
Bevestiging of verwerking van een BTW-suppletie.

BETALINGSHERINNERING
Herinnering over een openstaand bedrag.

AANMANING
Laatste waarschuwing voordat verdere invordering plaatsvindt.

DWANGBEVEL
Gedwongen invordering of inzet van een deurwaarder.

INFORMATIEVERZOEK
De Belastingdienst vraagt documenten of aanvullende informatie.

BOEKENONDERZOEK
Aankondiging van een controle van de administratie.

BOETE_VERZUIM
Boete wegens een te late aangifte of betaling.

BOETE_VERGRIJP
Boete wegens opzet of grove schuld.

IB_VOORLOPIG
Voorlopige aanslag inkomstenbelasting.

IB_DEFINITIEF
Definitieve aanslag inkomstenbelasting.

IB_TERUGGAAF
Teruggaaf inkomstenbelasting.

KOR_BESCHIKKING
Beslissing over de Kleineondernemersregeling.

NIET_BELASTINGDIENST
De afzender is niet de Nederlandse Belastingdienst.

ONBEKEND
De brief kan niet betrouwbaar worden geclassificeerd.

BELANGRIJKE REGELS:
- "Artikel 17" in combinatie met "Wet op de omzetbelasting" wijst op
  BTW_AMBTSHALVE.
- "Artikel 65" in combinatie met "AWR" wijst meestal op BTW_AMBTSHALVE.
- Een koptekst zoals "Uitspraak op bezwaarschrift" is niet genoeg om te bepalen
  dat de ontvanger zelf bezwaar heeft gemaakt.
- Gebruik BTW_BEZWAAR_UITSPRAAK alleen als de brief expliciet verwijst naar een
  door de ontvanger ingediend bezwaar.
- DWANGBEVEL, BOEKENONDERZOEK, BOETE_VERGRIJP en AANMANING hebben urgentie ROOD.
- INFORMATIEVERZOEK heeft urgentie ORANJE, of ROOD bij een korte deadline.
- Als de afzender geen Belastingdienst is, gebruik NIET_BELASTINGDIENST.

Retourneer exact deze structuur:
{
  "type": "<TYPE>",
  "afzender": "<naam afzender>",
  "periode": "<periode of null>",
  "dagtekening": "<datum of null>",
  "heeft_deadline": true,
  "deadline_datum": "<datum of null>",
  "actie_vereist": true,
  "urgentie": "GROEN",
  "zelf_ingediend": false,
  "kor_uitkomst": null
}
`.trim();

const EXPLANATION_SYSTEM_PROMPT = `
Je legt Nederlandse belastingbrieven uit aan ZZP'ers.
Schrijf in eenvoudig Nederlands op A2/B1-niveau.

REGELS:
- Gebruik maximaal 200 woorden.
- Gebruik korte, actieve zinnen.
- Schrijf geen financiële bedragen over; verwijs naar "het bedrag op je brief".
- Maak geen aannames over informatie die niet in de classificatie of brief staat.
- Gebruik de classificatie als leidend gegeven.
- Verwijs bij onzekerheid naar een boekhouder of de Belastingdienst.
- Maak duidelijk of en wanneer actie nodig is.
- Gebruik geen onnodig juridisch jargon.

Gebruik exact deze structuur:

📋 Wat is dit?
Een korte beschrijving van het soort brief.

📌 Waarom ontvangen?
Een eenvoudige uitleg waarom iemand deze brief ontvangt.

✅ Wat moet je doen?
Maximaal drie concrete, genummerde stappen.

⏰ Deadline
Noem de deadline uit de classificatie. Als die onbekend is, zeg dat de lezer
de datum op de brief moet controleren.

🚦 Urgentie
Noem de urgentie en leg die in één korte zin uit.
`.trim();

/**
 * Low-level helper for calling Claude's Messages API.
 *
 * In production, do not expose the API key in browser code.
 * Call Claude from a trusted server or serverless function.
 */
async function callClaude(params: {
  apiKey: string;
  model: string;
  maxTokens: number;
  system?: string;
  content: unknown;
  timeoutMs: number;
}): Promise<ClaudeResponse> {
  if (!params.apiKey) {
    throw new Error("Claude API key has not been configured.");
  }

  const response = await fetch(ANTHROPIC_API_URL, {
    method: "POST",
    signal: AbortSignal.timeout(params.timeoutMs),
    headers: {
      "x-api-key": params.apiKey,
      "anthropic-version": ANTHROPIC_API_VERSION,
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: params.model,
      max_tokens: params.maxTokens,
      ...(params.system ? { system: params.system } : {}),
      messages: [
        {
          role: "user",
          content: params.content,
        },
      ],
    }),
  });

  if (!response.ok) {
    // Avoid logging the letter body or API credentials.
    throw new Error(`Claude request failed with status ${response.status}.`);
  }

  return (await response.json()) as ClaudeResponse;
}

function getResponseText(response: ClaudeResponse): string {
  return (
    response.content
      ?.filter((block): block is ClaudeTextBlock => block.type === "text")
      .map((block) => block.text)
      .join("\n")
      .trim() ?? ""
  );
}

/**
 * Stage 1: OCR
 *
 * Claude Sonnet receives the image and returns only its raw text.
 */
async function extractLetterText(params: {
  imageBase64: string;
  mimeType: ImageMimeType;
  apiKey: string;
}): Promise<string> {
  const response = await callClaude({
    apiKey: params.apiKey,
    model: OCR_MODEL,
    maxTokens: 2_000,
    timeoutMs: 45_000,
    content: [
      {
        type: "image",
        source: {
          type: "base64",
          media_type: params.mimeType,
          data: params.imageBase64,
        },
      },
      {
        type: "text",
        text: [
          "Extraheer alle tekst uit deze brief.",
          "Geef alleen de ruwe tekst terug.",
          "Voeg geen uitleg, samenvatting of markdown toe.",
        ].join(" "),
      },
    ],
  });

  const extractedText = getResponseText(response);

  if (extractedText.length < 100) {
    throw new Error(
      "The image did not contain enough readable text for a reliable analysis.",
    );
  }

  return extractedText;
}

function fallbackClassification(): LetterClassification {
  return {
    type: "ONBEKEND",
    afzender: "Onbekend",
    periode: null,
    dagtekening: null,
    heeft_deadline: false,
    deadline_datum: null,
    actie_vereist: true,
    urgentie: "ORANJE",
    zelf_ingediend: false,
    kor_uitkomst: null,
  };
}

function parseClassification(rawResponse: string): LetterClassification {
  const cleaned = rawResponse
    .replace(/```json\s*/gi, "")
    .replace(/```\s*/g, "")
    .trim();

  try {
    const parsed = JSON.parse(cleaned) as Partial<LetterClassification>;

    if (typeof parsed.type !== "string") {
      throw new Error("Classification does not contain a type.");
    }

    const classification: LetterClassification = {
      ...fallbackClassification(),
      ...parsed,
    };

    /*
     * This combination is contradictory: an objection ruling should only be
     * used when the recipient actually submitted an objection.
     */
    if (
      classification.type === "BTW_BEZWAAR_UITSPRAAK" &&
      classification.zelf_ingediend === false
    ) {
      classification.type = "BTW_AMBTSHALVE";
    }

    /*
     * Normalize this category because it represents a correction performed
     * by the tax authority, not an action initiated by the recipient.
     */
    if (classification.type === "BTW_AMBTSHALVE") {
      classification.urgentie = "GROEN";
      classification.actie_vereist = false;
      classification.zelf_ingediend = false;
    }

    return classification;
  } catch {
    /*
     * If Claude returned almost-valid JSON, preserve the type when possible
     * instead of discarding the entire classification.
     */
    const typeMatch = rawResponse.match(/"type"\s*:\s*"([^"]+)"/);
    return {
      ...fallbackClassification(),
      type: (typeMatch?.[1] as LetterType | undefined) ?? "ONBEKEND",
    };
  }
}

/**
 * Stage 2: Classification
 *
 * Claude Haiku is used because this step is structured and relatively small.
 */
async function classifyLetter(params: {
  letterText: string;
  apiKey: string;
}): Promise<LetterClassification> {
  const response = await callClaude({
    apiKey: params.apiKey,
    model: CLASSIFICATION_MODEL,
    maxTokens: 600,
    timeoutMs: 15_000,
    system: CLASSIFICATION_SYSTEM_PROMPT,
    content: `Brief:\n\n${params.letterText}`,
  });

  const rawClassification = getResponseText(response);

  if (!rawClassification) {
    throw new Error("Claude returned an empty classification.");
  }

  return parseClassification(rawClassification);
}

function buildExplanationInput(
  letterText: string,
  classification: LetterClassification,
): string {
  /*
   * For this category, template wording in the original letter can be
   * misleading. Generate the explanation from normalized metadata instead.
   */
  if (classification.type === "BTW_AMBTSHALVE") {
    return `
Geclassificeerde brief:
Type: BTW_AMBTSHALVE
Periode: ${classification.periode ?? "onbekend"}
Datum brief: ${classification.dagtekening ?? "onbekend"}
Zelf ingediend door ontvanger: NEE
Urgentie: GROEN

Gebruik uitsluitend deze classificatie. Maak geen aannames op basis van
mogelijk tegenstrijdige template-tekst uit de oorspronkelijke brief.
`.trim();
  }

  return `
Classificatie:
${JSON.stringify(classification, null, 2)}

Oorspronkelijke brief:
${letterText}

Leg de brief uit volgens de systeeminstructies. Vertrouw de classificatie als
de formulering van de brief en de classificatie elkaar tegenspreken.
`.trim();
}

/**
 * Stage 3: Explanation
 *
 * Claude Sonnet turns the structured classification and letter text into a
 * short, actionable explanation.
 */
async function explainLetter(params: {
  letterText: string;
  classification: LetterClassification;
  apiKey: string;
}): Promise<string> {
  if (params.classification.type === "NIET_BELASTINGDIENST") {
    return "Deze brief lijkt niet afkomstig te zijn van de Belastingdienst.";
  }

  const response = await callClaude({
    apiKey: params.apiKey,
    model: EXPLANATION_MODEL,
    maxTokens: 800,
    timeoutMs: 25_000,
    system: EXPLANATION_SYSTEM_PROMPT,
    content: buildExplanationInput(
      params.letterText,
      params.classification,
    ),
  });

  const explanation = getResponseText(response);

  if (!explanation) {
    throw new Error("Claude returned an empty explanation.");
  }

  return explanation;
}

/**
 * Analyze an image of a Dutch tax letter.
 *
 * The API key must be supplied explicitly by trusted server-side code.
 * Never call this function directly from browser or client-side code.
 */
export async function analyzeLetterImage(params: {
  imageBase64: string;
  mimeType: ImageMimeType;
  apiKey: string;
}): Promise<AnalysisResult> {
  if (!params.apiKey) {
    throw new Error("Claude API key is required.");
  }

  const extractedText = await extractLetterText({
    imageBase64: params.imageBase64,
    mimeType: params.mimeType,
    apiKey: params.apiKey,
  });

  const classification = await classifyLetter({
    letterText: extractedText,
    apiKey: params.apiKey,
  });

  const explanation = await explainLetter({
    letterText: extractedText,
    classification,
    apiKey: params.apiKey,
  });

  return {
    extractedText,
    classification,
    explanation,
  };
}

/**
 * Analyze text copied from a Dutch tax letter.
 *
 * The API key must be supplied explicitly by trusted server-side code.
 * Never call this function directly from browser or client-side code.
 */
export async function analyzeLetterText(params: {
  letterText: string;
  apiKey: string;
}): Promise<AnalysisResult> {
  if (!params.apiKey) {
    throw new Error("Claude API key is required.");
  }

  const normalizedText = params.letterText.trim();

  if (!normalizedText) {
    throw new Error("Letter text is required.");
  }

  const classification = await classifyLetter({
    letterText: normalizedText,
    apiKey: params.apiKey,
  });

  const explanation = await explainLetter({
    letterText: normalizedText,
    classification,
    apiKey: params.apiKey,
  });

  return {
    extractedText: normalizedText,
    classification,
    explanation,
  };
}

/**
 * Example server-side usage:
 *
 * const imageResult = await analyzeLetterImage({
 *   imageBase64,
 *   mimeType: "image/jpeg",
 *   apiKey, // Claude API key supplied securely by the server
 * });
 *
 * const textResult = await analyzeLetterText({
 *   letterText,
 *   apiKey, // Claude API key supplied securely by the server
 * });
 */
