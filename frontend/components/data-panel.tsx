'use client'

import { useEffect, useRef, useState } from 'react'
import {
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  FileText,
  Send,
  ShieldCheck,
  Thermometer,
  Zap,
  Cable,
  Cpu,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { ComponentIdentification } from '@/lib/api'
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
  { label: 'Operating Voltage', icon: <Zap className="size-3.5" /> },
  { label: 'Pinout Diagram', icon: <Cable className="size-3.5" /> },
  { label: 'Datasheet', icon: <FileText className="size-3.5" /> },
  { label: 'Temperature Range', icon: <Thermometer className="size-3.5" /> },
]

interface DataPanelProps {
  state: StageState
  identification: ComponentIdentification | null
  messages: ChatMessage[]
  isQuerying: boolean
  errorMessage: string | null
  onAskQuestion: (question: string) => Promise<void>
}

export function DataPanel({
  state,
  identification,
  messages,
  isQuerying,
  errorMessage,
  onAskQuestion,
}: DataPanelProps) {
  const [specsOpen, setSpecsOpen] = useState(false)
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

  const handleQuickAction = (action: string) => {
    setInputValue(`What is the ${action.toLowerCase()} for this component?`)
  }

  if (state === 'default') {
    return (
      <div className="flex min-h-0 flex-1 flex-col items-center justify-center p-8 text-center">
        <div className="flex size-16 items-center justify-center rounded-full bg-muted">
          <Cpu className="size-8 text-muted-foreground" />
        </div>
        <h3 className="mt-4 text-lg font-medium text-foreground">
          No Component Scanned
        </h3>
        <p className="mt-1 max-w-xs text-sm text-muted-foreground">
          Upload a component photo or choose a demo image to identify the
          hardware and ask specification questions.
        </p>
      </div>
    )
  }

  if (state === 'scanning') {
    return (
      <div className="flex min-h-0 flex-1 flex-col gap-6 p-4">
        <div className="animate-pulse space-y-4">
          <div className="flex items-center gap-4">
            <div className="size-14 rounded-xl bg-muted" />
            <div className="flex-1 space-y-2">
              <div className="h-4 w-32 rounded bg-muted" />
              <div className="h-3 w-24 rounded bg-muted" />
            </div>
            <div className="size-12 rounded-full bg-muted" />
          </div>
          <div className="flex gap-2">
            {[1, 2, 3, 4].map((item) => (
              <div key={item} className="h-8 w-24 rounded-full bg-muted" />
            ))}
          </div>
          <div className="space-y-2">
            <div className="h-4 w-full rounded bg-muted" />
            <div className="h-4 w-5/6 rounded bg-muted" />
            <div className="h-4 w-4/6 rounded bg-muted" />
          </div>
        </div>
      </div>
    )
  }

  const specs = buildSpecs(identification)

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <ScrollArea className="min-h-0 flex-1">
        <div className="space-y-4 p-4">
          {errorMessage && (
            <div className="rounded-2xl border border-amber-300 bg-amber-50 p-4 text-sm text-amber-950">
              <div className="flex items-start gap-3">
                <AlertTriangle className="mt-0.5 size-4 shrink-0" />
                <div>
                  <p className="font-medium">Backend response warning</p>
                  <p className="mt-1 text-amber-900">{errorMessage}</p>
                </div>
              </div>
            </div>
          )}

          <div className="rounded-2xl border bg-card p-4 shadow-sm">
            <div className="flex items-start gap-4">
              <div className="flex size-14 shrink-0 items-center justify-center rounded-xl bg-primary/10">
                <Cpu className="size-7 text-primary" />
              </div>
              <div className="min-w-0 flex-1">
                <p className="text-sm text-muted-foreground">
                  {identification?.manufacturer || 'Unknown manufacturer'}
                </p>
                <h3 className="truncate text-lg font-semibold text-foreground">
                  {identification?.model_number ||
                    identification?.part_number ||
                    identification?.component_type ||
                    'Component identified'}
                </h3>
                <p className="mt-1 text-sm text-muted-foreground">
                  {identification?.component_type || 'Type not confirmed yet'}
                </p>
              </div>
              <ConfidenceRing
                confidence={Math.round(
                  (identification?.confidence_score ?? 0) * 100,
                )}
              />
            </div>

            <Collapsible open={specsOpen} onOpenChange={setSpecsOpen}>
              <CollapsibleTrigger asChild>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-3 w-full justify-between text-muted-foreground hover:text-foreground"
                >
                  <span>Identification Details</span>
                  {specsOpen ? (
                    <ChevronUp className="size-4" />
                  ) : (
                    <ChevronDown className="size-4" />
                  )}
                </Button>
              </CollapsibleTrigger>
              <CollapsibleContent className="mt-2">
                <div className="grid grid-cols-2 gap-2">
                  {specs.map((spec) => (
                    <div
                      key={spec.label}
                      className="rounded-lg bg-muted/50 p-2.5"
                    >
                      <p className="text-xs text-muted-foreground">
                        {spec.label}
                      </p>
                      <p className="text-sm font-medium text-foreground">
                        {spec.value}
                      </p>
                    </div>
                  ))}
                </div>
              </CollapsibleContent>
            </Collapsible>
          </div>

          <ScrollArea className="w-full whitespace-nowrap">
            <div className="flex gap-2 pb-2">
              {quickActions.map((action) => (
                <button
                  key={action.label}
                  onClick={() => handleQuickAction(action.label)}
                  className="inline-flex items-center gap-1.5 rounded-full border bg-card px-3 py-1.5 text-sm font-medium text-foreground shadow-sm transition-all hover:border-primary/50 hover:bg-accent active:scale-95"
                  type="button"
                >
                  {action.icon}
                  {action.label}
                </button>
              ))}
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>

          <div className="space-y-4">
            {messages.map((message) => (
              <ChatMessageBubble key={message.id} message={message} />
            ))}
            {isQuerying && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="flex gap-1">
                  <span className="size-2 animate-bounce rounded-full bg-primary [animation-delay:-0.3s]" />
                  <span className="size-2 animate-bounce rounded-full bg-primary [animation-delay:-0.15s]" />
                  <span className="size-2 animate-bounce rounded-full bg-primary" />
                </div>
                <span>Retrieving documentation and drafting a cited answer...</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </ScrollArea>

      <div className="border-t bg-card p-4">
        <form
          onSubmit={(event) => {
            event.preventDefault()
            void handleSendMessage()
          }}
          className="flex items-center gap-2"
        >
          <Input
            value={inputValue}
            onChange={(event) => setInputValue(event.target.value)}
            placeholder="Ask about this component..."
            className="flex-1 rounded-xl"
          />
          <Button
            type="submit"
            size="icon"
            disabled={!inputValue.trim() || isQuerying}
            className="shrink-0 rounded-xl transition-transform active:scale-95"
          >
            <Send className="size-4" />
            <span className="sr-only">Send message</span>
          </Button>
        </form>
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
      { label: 'Part Number', value: 'Pending' },
      { label: 'Document Lookup', value: 'Pending' },
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
      label: 'Part Number',
      value: identification.part_number || 'Not extracted',
    },
    {
      label: 'Lookup Strategy',
      value: identification.should_attempt_document_lookup
        ? 'Documentation lookup enabled'
        : 'Manual follow-up recommended',
    },
    {
      label: 'Fallback Tier',
      value: identification.fallback_tier.toString(),
    },
    {
      label: 'OCR Text',
      value: identification.raw_ocr_text || 'No OCR text found',
    },
  ]
}

function ConfidenceRing({ confidence }: { confidence: number }) {
  const radius = 18
  const circumference = 2 * Math.PI * radius
  const clamped = Math.max(0, Math.min(confidence, 100))
  const strokeDashoffset = circumference - (clamped / 100) * circumference

  return (
    <div className="relative flex size-12 items-center justify-center">
      <svg className="size-12 -rotate-90" viewBox="0 0 40 40">
        <circle
          cx="20"
          cy="20"
          r={radius}
          strokeWidth="3"
          fill="none"
          className="stroke-muted"
        />
        <circle
          cx="20"
          cy="20"
          r={radius}
          strokeWidth="3"
          fill="none"
          strokeLinecap="round"
          className="stroke-green-500 transition-all duration-500"
          style={{
            strokeDasharray: circumference,
            strokeDashoffset,
          }}
        />
      </svg>
      <span className="absolute text-xs font-semibold text-foreground">
        {clamped}%
      </span>
    </div>
  )
}

function ChatMessageBubble({ message }: { message: ChatMessage }) {
  if (message.role === 'user') {
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl bg-primary px-4 py-2 text-sm text-primary-foreground">
          {message.content}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-3">
      <div className="rounded-2xl border bg-card p-4 shadow-sm">
        <SimpleMarkdownContent content={message.content} />
      </div>

      {message.citations && message.citations.length > 0 && (
        <ScrollArea className="w-full whitespace-nowrap">
          <div className="flex gap-2 pb-2">
            {message.citations.map((citation) => (
              <a
                key={citation.id}
                href={citation.url}
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-2 rounded-lg border bg-card px-3 py-2 text-sm shadow-sm transition-all hover:border-primary/50 hover:shadow-md"
              >
                <span className="flex size-8 items-center justify-center rounded-full bg-primary/10 text-primary">
                  <ShieldCheck className="size-4" />
                </span>
                <div className="flex min-w-0 flex-col">
                  <span className="max-w-48 truncate font-medium text-foreground">
                    {citation.title}
                  </span>
                  <span className="text-xs text-muted-foreground">
                    {citation.documentType}
                    {citation.pageNumber ? ` | page ${citation.pageNumber}` : ''}
                  </span>
                </div>
                <ExternalLink className="ml-1 size-3.5 text-muted-foreground" />
              </a>
            ))}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      )}
    </div>
  )
}

function SimpleMarkdownContent({ content }: { content: string }) {
  return (
    <div className="space-y-2">
      {content.split('\n').filter(Boolean).map((line, index) => (
        <p key={`${line}-${index}`} className="text-sm leading-relaxed text-foreground">
          {line}
        </p>
      ))}
    </div>
  )
}
