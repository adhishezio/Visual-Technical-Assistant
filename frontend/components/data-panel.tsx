'use client'

import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  Clock3,
  FileText,
  Gauge,
  Send,
  ShieldCheck,
  Sparkles,
  Zap,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { ComponentIdentification, QueryLogEntry } from '@/lib/api'
import type { StageState } from './camera-stage'

export interface SourceCitation {
  id: string
  title: string
  url: string
  documentType: string
  pageNumber?: number | null
}

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant'
  content: string
  citations?: SourceCitation[]
}

const quickActions = [
  {
    label: 'Operating voltage',
    prompt: 'What is the operating voltage for this component?',
    icon: <Zap className="size-3.5" />,
  },
  {
    label: 'Current rating',
    prompt: 'What is the rated current for this component?',
    icon: <Gauge className="size-3.5" />,
  },
  {
    label: 'Applicable standards',
    prompt: 'What standards or certifications apply to this component?',
    icon: <ShieldCheck className="size-3.5" />,
  },
  {
    label: 'Datasheet summary',
    prompt: 'Summarize the official datasheet for this component.',
    icon: <FileText className="size-3.5" />,
  },
]

interface DataPanelProps {
  state: StageState
  identification: ComponentIdentification | null
  messages: ChatMessage[]
  historyEntries: QueryLogEntry[]
  historyLoading: boolean
  isQuerying: boolean
  errorMessage: string | null
  onAskQuestion: (question: string) => Promise<void>
}

export function DataPanel({
  state,
  identification,
  messages,
  historyEntries,
  historyLoading,
  isQuerying,
  errorMessage,
  onAskQuestion,
}: DataPanelProps) {
  const [detailsOpen, setDetailsOpen] = useState(true)
  const [historyOpen, setHistoryOpen] = useState(false)
  const [inputValue, setInputValue] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isQuerying])

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isQuerying) return
    const question = inputValue.trim()
    setInputValue('')
    await onAskQuestion(question)
  }

  if (state === 'default') {
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col items-center justify-center px-6 py-10 text-center">
        <div className="rounded-[1.75rem] border border-slate-200 bg-white/92 px-8 py-10 shadow-[0_24px_70px_-44px_rgba(15,23,42,0.45)]">
          <div className="mx-auto flex size-16 items-center justify-center rounded-2xl bg-sky-50 text-primary">
            <Sparkles className="size-8" />
          </div>
          <h3 className="mt-5 text-xl font-semibold text-slate-900">
            Inspection workspace
          </h3>
          <p className="mt-3 max-w-sm text-sm leading-7 text-slate-600">
            Select a real component image to open the analysis workspace. The
            right panel will track identification confidence, documentation
            readiness, citations, and question history.
          </p>
        </div>
      </div>
    )
  }

  if (state === 'scanning') {
    return (
      <div className="flex min-h-0 min-w-0 flex-1 flex-col gap-4 p-5">
        <SkeletonCard />
        <SkeletonCard />
      </div>
    )
  }

  const specs = buildSpecs(identification)
  const confidence = Math.round((identification?.confidence_score ?? 0) * 100)

  return (
    <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden bg-[linear-gradient(180deg,rgba(255,255,255,0.92),rgba(239,243,247,0.96))]">
      <div className="min-w-0 border-b border-slate-200/90 bg-white/80 px-5 py-4 backdrop-blur">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              Inspection summary
            </p>
            <h2 className="mt-1 break-words text-2xl font-semibold tracking-tight text-slate-900">
              {identification?.model_number ||
                identification?.part_number ||
                identification?.component_type ||
                'Component identified'}
            </h2>
            <p className="mt-2 break-words text-sm leading-6 text-slate-600">
              {identification?.manufacturer || 'Unknown manufacturer'}
              {identification?.component_type
                ? ` | ${identification.component_type}`
                : ''}
            </p>
          </div>

          <div className="rounded-2xl border border-slate-200 bg-slate-50 px-4 py-3 text-right shadow-sm">
            <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
              Confidence
            </p>
            <p className="mt-1 text-2xl font-semibold text-slate-900">
              {confidence}%
            </p>
          </div>
        </div>

        <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-200">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              confidence >= 80
                ? 'bg-emerald-500'
                : confidence >= 60
                  ? 'bg-amber-500'
                  : 'bg-rose-500'
            }`}
            style={{ width: `${Math.max(confidence, 4)}%` }}
          />
        </div>

        <div className="mt-4 flex flex-wrap gap-2">
          <StatusPill
            tone={identification?.should_attempt_document_lookup ? 'good' : 'muted'}
          >
            {identification?.should_attempt_document_lookup
              ? 'Documentation cache warming'
              : 'Manual follow-up recommended'}
          </StatusPill>
          <StatusPill tone="muted">Chroma cache</StatusPill>
          <StatusPill tone="accent">Citations required</StatusPill>
        </div>
      </div>

      <div className="flex min-h-0 min-w-0 flex-1 flex-col overflow-hidden">
        <div className="min-w-0 border-b border-slate-200/80 bg-white/70 px-5 py-4">
          {errorMessage && (
            <div className="mb-4 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <div className="min-w-0">
                  <p className="font-semibold">Backend response warning</p>
                  <p className="mt-1 break-words leading-6 text-amber-900">
                    {errorMessage}
                  </p>
                </div>
              </div>
            </div>
          )}

          <Collapsible open={detailsOpen} onOpenChange={setDetailsOpen}>
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="flex w-full items-center justify-between rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-left transition-colors hover:border-slate-300 hover:bg-white"
              >
                <div>
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                    Identification details
                  </p>
                  <p className="mt-1 text-sm text-slate-600">
                    Manufacturer, part data, and current lookup posture
                  </p>
                </div>
                <span className="text-sm font-medium text-slate-600">
                  {detailsOpen ? 'Hide' : 'Show'}
                </span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <div className="grid gap-3 sm:grid-cols-2">
                {specs.map((spec) => (
                  <div
                    key={spec.label}
                    className="rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm"
                  >
                    <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-slate-500">
                      {spec.label}
                    </p>
                    <p className="mt-2 break-words font-mono text-sm leading-6 text-slate-900">
                      {spec.value}
                    </p>
                  </div>
                ))}
              </div>
            </CollapsibleContent>
          </Collapsible>

          <Collapsible open={historyOpen} onOpenChange={setHistoryOpen}>
            <CollapsibleTrigger asChild>
              <button
                type="button"
                className="mt-4 flex w-full items-center justify-between rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-left transition-colors hover:border-slate-300 hover:bg-white"
              >
                <div className="min-w-0">
                  <p className="text-xs font-semibold uppercase tracking-[0.18em] text-slate-500">
                    Maintenance history
                  </p>
                  <p className="mt-1 break-words text-sm text-slate-600">
                    {buildHistorySummary(historyEntries, historyLoading)}
                  </p>
                </div>
                <span className="inline-flex items-center gap-2 text-sm font-medium text-slate-600">
                  <Clock3 className="size-4" />
                  {historyOpen ? 'Hide' : 'Show'}
                </span>
              </button>
            </CollapsibleTrigger>
            <CollapsibleContent className="mt-3">
              <div className="space-y-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm">
                {historyLoading ? (
                  <p className="text-sm text-slate-500">
                    Loading previous questions for this component...
                  </p>
                ) : historyEntries.length > 0 ? (
                  historyEntries.map((entry) => (
                    <div
                      key={entry.id ?? `${entry.timestamp}-${entry.question}`}
                      className="rounded-2xl border border-slate-200 bg-slate-50/70 px-4 py-3"
                    >
                      <div className="flex flex-wrap items-center justify-between gap-2">
                        <p className="text-sm font-semibold text-slate-900">
                          {entry.question}
                        </p>
                        <span className="text-xs uppercase tracking-[0.16em] text-slate-500">
                          {formatHistoryTimestamp(entry.timestamp)}
                        </span>
                      </div>
                      <p className="mt-2 text-sm leading-6 text-slate-600">
                        {entry.answer}
                      </p>
                      <p className="mt-2 text-[11px] uppercase tracking-[0.18em] text-slate-500">
                        {entry.source}
                        {entry.doc_source ? ` | ${entry.doc_source}` : ''}
                      </p>
                    </div>
                  ))
                ) : (
                  <p className="text-sm text-slate-500">
                    No component-specific maintenance history yet for this scan.
                  </p>
                )}
              </div>
            </CollapsibleContent>
          </Collapsible>

          <div className="mt-4 flex flex-wrap gap-2">
            {quickActions.map((action) => (
              <button
                key={action.label}
                onClick={() => setInputValue(action.prompt)}
                className="inline-flex items-center gap-2 rounded-full border border-slate-200 bg-white px-3.5 py-2 text-sm font-medium text-slate-700 shadow-sm transition-all hover:border-primary/35 hover:bg-sky-50 hover:text-slate-900"
                type="button"
              >
                {action.icon}
                {action.label}
              </button>
            ))}
          </div>
        </div>

        <div className="min-h-0 min-w-0 flex-1 overflow-hidden">
          <div className="flex h-full min-w-0 flex-col overflow-hidden">
            <div className="min-h-0 min-w-0 flex-1 overflow-y-auto px-5 py-5">
              <div className="space-y-4">
                {messages.map((message) => (
                  <ChatMessageBubble key={message.id} message={message} />
                ))}
                {isQuerying && (
                  <div className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-sm text-slate-600 shadow-sm">
                    <div className="flex items-center gap-2">
                      <div className="flex gap-1">
                        <span className="size-2 animate-bounce rounded-full bg-primary [animation-delay:-0.3s]" />
                        <span className="size-2 animate-bounce rounded-full bg-primary [animation-delay:-0.15s]" />
                        <span className="size-2 animate-bounce rounded-full bg-primary" />
                      </div>
                      Preparing a cited answer from official documentation...
                    </div>
                  </div>
                )}
                <div ref={messagesEndRef} />
              </div>
            </div>

            <div className="border-t border-slate-200/90 bg-white/90 px-5 py-4 backdrop-blur">
              <form
                onSubmit={(event) => {
                  event.preventDefault()
                  void handleSendMessage()
                }}
                className="space-y-3"
              >
                <div className="rounded-[1.75rem] border border-slate-300 bg-slate-50 p-2 shadow-[0_16px_32px_-24px_rgba(15,23,42,0.6)]">
                  <div className="flex min-w-0 items-center gap-2">
                    <Input
                      value={inputValue}
                      onChange={(event) => setInputValue(event.target.value)}
                      placeholder="Ask for voltage, standards, manuals, wiring notes, or maintenance specs..."
                      className="h-12 flex-1 border-0 bg-transparent px-4 text-sm shadow-none focus-visible:ring-0"
                    />
                    <Button
                      type="submit"
                      size="icon"
                      disabled={!inputValue.trim() || isQuerying}
                      className="size-11 shrink-0 rounded-full shadow-[0_14px_30px_-20px_rgba(37,99,235,0.85)]"
                    >
                      <Send className="size-4" />
                      <span className="sr-only">Send message</span>
                    </Button>
                  </div>
                </div>
                <p className="text-xs uppercase tracking-[0.18em] text-slate-500">
                  The assistant will refuse uncited answers unless the reply is
                  explicitly grounded in image identification only.
                </p>
              </form>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

function buildSpecs(
  identification: ComponentIdentification | null,
): Array<{ label: string; value: string }> {
  if (!identification) {
    return [
      { label: 'Manufacturer', value: 'Pending' },
      { label: 'Model', value: 'Pending' },
      { label: 'Part number', value: 'Pending' },
      { label: 'Lookup strategy', value: 'Pending' },
    ]
  }

  return [
    {
      label: 'Manufacturer',
      value: identification.manufacturer || 'Unknown',
    },
    {
      label: 'Model',
      value: identification.model_number || 'Unknown',
    },
    {
      label: 'Part number',
      value: identification.part_number || 'Not extracted',
    },
    {
      label: 'Component serial',
      value: identification.component_serial || 'Not available',
    },
    {
      label: 'Lookup strategy',
      value: identification.should_attempt_document_lookup
        ? 'Background cache warm-up enabled'
        : 'Documentation lookup not yet trusted',
    },
    {
      label: 'Fallback tier',
      value: identification.fallback_tier.toString(),
    },
    {
      label: 'OCR / extracted text',
      value: identification.raw_ocr_text || 'No OCR text found',
    },
  ]
}

function buildHistorySummary(entries: QueryLogEntry[], isLoading: boolean): string {
  if (isLoading) {
    return 'Loading previous questions for this component.'
  }
  if (entries.length === 0) {
    return 'No prior maintenance history recorded for this component yet.'
  }
  return `${entries[0].component_model} - last queried ${formatRelativeTimestamp(entries[0].timestamp)}`
}

function formatHistoryTimestamp(value: string): string {
  return new Intl.DateTimeFormat('en', {
    month: 'short',
    day: 'numeric',
  }).format(new Date(value))
}

function formatRelativeTimestamp(value: string): string {
  const date = new Date(value)
  const diffMs = Date.now() - date.getTime()
  const day = 24 * 60 * 60 * 1000

  if (diffMs < day) {
    return 'today'
  }
  if (diffMs < day * 2) {
    return 'yesterday'
  }
  if (diffMs < day * 30) {
    return `${Math.max(1, Math.floor(diffMs / day))} days ago`
  }
  if (diffMs < day * 365) {
    return `${Math.max(1, Math.floor(diffMs / (day * 30)))} months ago`
  }
  return `${Math.max(1, Math.floor(diffMs / (day * 365)))} years ago`
}

function StatusPill({
  children,
  tone,
}: {
  children: string
  tone: 'good' | 'muted' | 'accent'
}) {
  const toneClass =
    tone === 'good'
      ? 'border-emerald-200 bg-emerald-50 text-emerald-800'
      : tone === 'accent'
        ? 'border-sky-200 bg-sky-50 text-sky-800'
        : 'border-slate-200 bg-slate-100 text-slate-600'

  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${toneClass}`}
    >
      {children}
    </span>
  )
}

function SkeletonCard() {
  return (
    <div className="animate-pulse rounded-[1.75rem] border border-slate-200 bg-white/85 p-5 shadow-sm">
      <div className="h-4 w-32 rounded bg-slate-200" />
      <div className="mt-4 h-10 w-full rounded-xl bg-slate-100" />
      <div className="mt-3 h-10 w-5/6 rounded-xl bg-slate-100" />
      <div className="mt-3 h-24 w-full rounded-2xl bg-slate-100" />
    </div>
  )
}

function ChatMessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-full rounded-[1.75rem] bg-primary px-4 py-3 text-sm leading-7 text-primary-foreground shadow-[0_18px_36px_-24px_rgba(37,99,235,0.8)]">
          <p className="break-words">{message.content}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="rounded-[1.75rem] border border-slate-200 bg-white px-4 py-4 shadow-sm">
        <SimpleMarkdownContent content={message.content} />
      </div>

      {message.citations && message.citations.length > 0 && (
        <div className="space-y-2">
          {message.citations.map((citation) => (
            <a
              key={citation.id}
              href={citation.url}
              target="_blank"
              rel="noreferrer"
              className="flex min-w-0 items-start gap-3 rounded-2xl border border-slate-200 bg-slate-50/80 px-4 py-3 text-sm shadow-sm transition-colors hover:border-primary/35 hover:bg-white"
            >
              <span className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-full bg-sky-50 text-primary">
                <ShieldCheck className="size-4" />
              </span>
              <div className="min-w-0">
                <p className="truncate font-semibold text-slate-900">
                  {citation.title}
                </p>
                <p className="mt-1 break-words text-xs uppercase tracking-[0.18em] text-slate-500">
                  {citation.documentType}
                  {citation.pageNumber ? ` | page ${citation.pageNumber}` : ''}
                </p>
                <p className="mt-2 break-all text-xs text-slate-500">
                  {citation.url}
                </p>
              </div>
            </a>
          ))}
        </div>
      )}
    </div>
  )
}

function SimpleMarkdownContent({ content }: { content: string }) {
  return (
    <div className="space-y-2">
      {content
        .split('\n')
        .filter(Boolean)
        .map((line, index) => (
          <p
            key={`${line}-${index}`}
            className="break-words text-sm leading-7 text-slate-700"
          >
            {line}
          </p>
        ))}
    </div>
  )
}
