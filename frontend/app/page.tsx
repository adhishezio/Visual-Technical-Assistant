'use client'

import { startTransition, useEffect, useState } from 'react'
import { Header } from '@/components/header'
import {
  CameraStage,
  type SelectedImage,
  type StageState,
} from '@/components/camera-stage'
import {
  DataPanel,
  type ChatMessage,
  type SourceCitation,
} from '@/components/data-panel'
import {
  getComponentHistory,
  identifyComponent,
  queryComponent,
  type AnswerWithCitations,
  type ComponentIdentification,
  type QueryLogEntry,
} from '@/lib/api'

function revokeIfBlobUrl(url: string | null | undefined) {
  if (url?.startsWith('blob:')) {
    URL.revokeObjectURL(url)
  }
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error && error.message) {
    return error.message
  }
  return 'The request failed. Please try again with a clearer image.'
}

function buildIdentificationMessage(
  identification: ComponentIdentification,
): ChatMessage {
  const componentLabel =
    identification.model_number ||
    identification.part_number ||
    identification.component_type ||
    'component'
  const confidencePercent = Math.round(identification.confidence_score * 100)

  const lines = [
    `Identified ${componentLabel}${identification.manufacturer ? ` from ${identification.manufacturer}` : ''}.`,
    `Confidence: ${confidencePercent}%.`,
  ]

  if (identification.part_number) {
    lines.push(`Part number: ${identification.part_number}.`)
  }

  if (identification.visual_description) {
    lines.push(identification.visual_description)
  }

  if (!identification.part_number && identification.model_number) {
    lines.push(
      'Confidence is lower because the system matched the manufacturer and model, but it did not confirm an exact part number from the label.',
    )
  }

  if (identification.should_attempt_document_lookup) {
    lines.push(
      'Official documentation is being prepared in the background. Ask a specification or maintenance question and the system will answer from cited sources.',
    )
  } else {
    lines.push(
      'Documentation lookup is not reliable yet for this image. Try a clearer label shot or ask after re-capturing the part number.',
    )
  }

  return {
    id: `assistant-identification-${Date.now()}`,
    role: 'assistant',
    content: lines.join('\n'),
  }
}

function buildCitationTitle(citation: SourceCitation): string {
  return citation.title
}

function mapAnswerCitations(answer: AnswerWithCitations): SourceCitation[] {
  const citationMap = new Map<string, SourceCitation>()

  for (const citation of answer.citations) {
    const metadata = citation.chunk.metadata
    const url = metadata.source_url
    const pageNumber = citation.chunk.page_number ?? null
    const key = `${url}::${pageNumber ?? 'na'}`
    if (citationMap.has(key)) continue

    let fallbackTitle = url
    try {
      fallbackTitle = new URL(url).hostname.replace(/^www\./, '')
    } catch {
      fallbackTitle = url
    }

    citationMap.set(key, {
      id: key,
      title: metadata.source_title || fallbackTitle,
      url,
      documentType: metadata.document_type,
      pageNumber,
    })
  }

  return [...citationMap.values()]
}

function buildAnswerMessage(answer: AnswerWithCitations): ChatMessage {
  const citations = mapAnswerCitations(answer).map((citation) => ({
    ...citation,
    title: buildCitationTitle(citation),
  }))

  return {
    id: `assistant-answer-${Date.now()}`,
    role: 'assistant',
    content: answer.answer_text,
    citations,
  }
}

export default function HomePage() {
  const [stageState, setStageState] = useState<StageState>('default')
  const [selectedImage, setSelectedImage] = useState<SelectedImage | null>(null)
  const [identification, setIdentification] =
    useState<ComponentIdentification | null>(null)
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [historyEntries, setHistoryEntries] = useState<QueryLogEntry[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [isQuerying, setIsQuerying] = useState(false)

  useEffect(() => {
    return () => {
      revokeIfBlobUrl(selectedImage?.previewUrl)
    }
  }, [selectedImage])

  const handleResetWorkspace = () => {
    revokeIfBlobUrl(selectedImage?.previewUrl)
    startTransition(() => {
      setStageState('default')
      setSelectedImage(null)
      setIdentification(null)
      setMessages([])
      setHistoryEntries([])
      setHistoryLoading(false)
      setErrorMessage(null)
      setIsQuerying(false)
    })
  }

  const handleImageSelected = async (selection: SelectedImage) => {
    revokeIfBlobUrl(selectedImage?.previewUrl)
    setSelectedImage(selection)
    setStageState('scanning')
    setIdentification(null)
    setMessages([])
    setHistoryEntries([])
    setHistoryLoading(false)
    setErrorMessage(null)

    try {
      const result = await identifyComponent(selection.file)
      startTransition(() => {
        setIdentification(result)
        setMessages([buildIdentificationMessage(result)])
      })

      if (result.component_serial) {
        setHistoryLoading(true)
        try {
          const history = await getComponentHistory(result.component_serial)
          startTransition(() => {
            setHistoryEntries(history)
          })
        } catch {
          startTransition(() => {
            setHistoryEntries([])
          })
        } finally {
          setHistoryLoading(false)
        }
      }
    } catch (error) {
      setErrorMessage(getErrorMessage(error))
      setMessages([
        {
          id: `assistant-identification-error-${Date.now()}`,
          role: 'assistant',
          content:
            'The backend could not complete identification for this image. Try a clearer close-up or use a label-facing photo.',
        },
      ])
    } finally {
      setStageState('active')
    }
  }

  const handleAskQuestion = async (question: string) => {
    if (!selectedImage) {
      setErrorMessage('Select an image before asking a question.')
      return
    }

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      role: 'user',
      content: question,
    }

    setMessages((current) => [...current, userMessage])
    setIsQuerying(true)
    setErrorMessage(null)

    try {
      const answer = await queryComponent(
        selectedImage.file,
        question,
        identification,
      )
      setMessages((current) => [...current, buildAnswerMessage(answer)])

      if (identification?.component_serial) {
        try {
          const history = await getComponentHistory(identification.component_serial)
          startTransition(() => {
            setHistoryEntries(history)
          })
        } catch {
          // Keep the previous history list if refresh fails.
        }
      }
    } catch (error) {
      const message = getErrorMessage(error)
      setErrorMessage(message)
      setMessages((current) => [
        ...current,
        {
          id: `assistant-query-error-${Date.now()}`,
          role: 'assistant',
          content:
            'The backend could not return a cited answer for that question. Try again with a more specific question or re-capture the label.',
        },
      ])
    } finally {
      setIsQuerying(false)
    }
  }

  return (
    <div className="industrial-shell flex h-svh min-w-0 flex-col overflow-hidden bg-background">
      <Header
        hasActiveInspection={stageState !== 'default'}
        onReset={handleResetWorkspace}
      />

      <main className="grid min-h-0 min-w-0 flex-1 grid-cols-1 overflow-hidden xl:grid-cols-[minmax(0,1fr)_minmax(21rem,28rem)] 2xl:grid-cols-[minmax(0,1.08fr)_minmax(23rem,30rem)]">
        <CameraStage
          state={stageState}
          selectedImage={selectedImage}
          onImageSelected={handleImageSelected}
          onReset={handleResetWorkspace}
        />

        <div className="flex min-h-0 min-w-0 flex-1 flex-col border-t border-slate-300/80 bg-white/88 backdrop-blur xl:border-l xl:border-t-0">
          <DataPanel
            state={stageState}
            identification={identification}
            messages={messages}
            historyEntries={historyEntries}
            historyLoading={historyLoading}
            isQuerying={isQuerying}
            errorMessage={errorMessage}
            onAskQuestion={handleAskQuestion}
          />
        </div>
      </main>
    </div>
  )
}
