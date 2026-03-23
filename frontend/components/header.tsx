'use client'

import { ArrowLeft, Cpu, DatabaseZap, ShieldCheck } from 'lucide-react'
import { Button } from '@/components/ui/button'

interface HeaderProps {
  hasActiveInspection: boolean
  onReset: () => void
}

function StatusBadge({
  label,
  tone = 'default',
}: {
  label: string
  tone?: 'default' | 'accent'
}) {
  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.18em] ${
        tone === 'accent'
          ? 'border-amber-300/80 bg-amber-50 text-amber-900'
          : 'border-slate-300/80 bg-white/85 text-slate-600'
      }`}
    >
      {label}
    </span>
  )
}

export function Header({ hasActiveInspection, onReset }: HeaderProps) {
  return (
    <header className="z-40 border-b border-slate-300/80 bg-white/88 backdrop-blur-xl">
      <div className="flex min-w-0 flex-wrap items-center justify-between gap-4 px-4 py-4 sm:px-6">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-11 shrink-0 items-center justify-center rounded-2xl bg-primary shadow-[0_14px_30px_-18px_rgba(37,99,235,0.8)]">
            <Cpu className="size-5 text-primary-foreground" />
          </div>
          <div className="min-w-0">
            <p className="text-[11px] font-semibold uppercase tracking-[0.22em] text-slate-500">
              Industrial Diagnostics
            </p>
            <h1 className="truncate text-2xl font-semibold tracking-tight text-slate-900">
              Visual Technical Assistant
            </h1>
          </div>
        </div>

        <div className="flex flex-wrap items-center justify-end gap-2">
          <StatusBadge label="Chroma Cache" />
          <StatusBadge label="Citation Guard" tone="accent" />
          <span className="hidden items-center gap-2 rounded-full border border-slate-300/80 bg-slate-50/90 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-slate-600 lg:inline-flex">
            <DatabaseZap className="size-3.5" />
            No Vertex Active
          </span>
          <span className="hidden items-center gap-2 rounded-full border border-emerald-200/80 bg-emerald-50/90 px-3 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-emerald-800 md:inline-flex">
            <ShieldCheck className="size-3.5" />
            Official Sources Only
          </span>
          {hasActiveInspection && (
            <Button
              type="button"
              variant="outline"
              className="rounded-full border-slate-300 bg-white/90 px-4 text-slate-700 shadow-sm"
              onClick={onReset}
            >
              <ArrowLeft className="size-4" />
              New inspection
            </Button>
          )}
        </div>
      </div>
    </header>
  )
}
