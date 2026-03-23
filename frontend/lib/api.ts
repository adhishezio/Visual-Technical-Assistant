export interface OCRResult {
  raw_text: string
  average_confidence: number
  detected_part_numbers: string[]
  provider?: string | null
}

export interface ComponentIdentification {
  manufacturer?: string | null
  model_number?: string | null
  part_number?: string | null
  component_type?: string | null
  confidence_score: number
  fallback_tier: number
  raw_ocr_text: string
  visual_description?: string | null
  ocr_result?: OCRResult | null
  should_attempt_document_lookup: boolean
  requires_manual_input: boolean
  error_details?: string | null
}

export interface DocumentMetadata {
  source_url: string
  source_title?: string | null
  manufacturer?: string | null
  model_number?: string | null
  part_number?: string | null
  document_type: string
  revision?: string | null
  page_map: Record<string, number>
  retrieved_at: string
  content_hash: string
  cache_key: {
    value: string
  }
}

export interface DocumentChunk {
  chunk_text: string
  chunk_index: number
  metadata: DocumentMetadata
  page_number?: number | null
  section_title?: string | null
}

export interface RetrievedChunk {
  chunk: DocumentChunk
  similarity_score: number
}

export interface AnswerWithCitations {
  answer_text: string
  citations: RetrievedChunk[]
  confidence: number
  has_citations: boolean
}

function getApiBaseUrl(): string {
  const apiUrl = process.env.NEXT_PUBLIC_API_URL
  if (!apiUrl) {
    throw new Error(
      "NEXT_PUBLIC_API_URL is not configured. Add it to frontend/.env.local.",
    )
  }
  return apiUrl.replace(/\/+$/, "")
}

async function parseJsonResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const body = await response.text()
    throw new Error(body || `Request failed with status ${response.status}`)
  }
  return (await response.json()) as T
}

export async function identifyComponent(imageFile: File): Promise<ComponentIdentification> {
  const formData = new FormData()
  formData.append("image", imageFile)

  const response = await fetch(`${getApiBaseUrl()}/identify`, {
    method: "POST",
    body: formData,
  })

  return parseJsonResponse<ComponentIdentification>(response)
}

export async function queryComponent(
  imageFile: File,
  question: string,
): Promise<AnswerWithCitations> {
  const formData = new FormData()
  formData.append("image", imageFile)
  formData.append("question", question)

  const response = await fetch(`${getApiBaseUrl()}/query`, {
    method: "POST",
    body: formData,
  })

  return parseJsonResponse<AnswerWithCitations>(response)
}
