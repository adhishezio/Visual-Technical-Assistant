'use client'

import { useRef, useState, type ChangeEvent, type ReactNode } from 'react'
import {
  ArrowLeft,
  Camera,
  ChevronRight,
  Cpu,
  Factory,
  FileSearch,
  ScanSearch,
  Upload,
} from 'lucide-react'
import { Button } from '@/components/ui/button'

export type StageState = 'default' | 'active' | 'scanning'

export interface SelectedImage {
  id: string
  name: string
  previewUrl: string
  file: File
}

interface DemoComponent {
  id: string
  name: string
  description: string
  icon: ReactNode
  imagePath: string
}

const demoComponents: DemoComponent[] = [
  {
    id: 'abb-rcbo',
    name: 'ABB RCBO',
    description: 'Residual current breaker with integrated protection',
    icon: <Factory className="size-4" />,
    imagePath: '/demo-abb-rcbo.jpg',
  },
  {
    id: 'siemens-label',
    name: 'Siemens Label',
    description: 'Industrial module packaging label with part number',
    icon: <FileSearch className="size-4" />,
    imagePath: '/demo-siemens-label.jpg',
  },
  {
    id: 'abb-motor-plate',
    name: 'ABB Motor Plate',
    description: 'Annotated motor nameplate for spec extraction',
    icon: <Cpu className="size-4" />,
    imagePath: '/demo-abb-motor-plate.jpg',
  },
  {
    id: 'abb-breaker',
    name: 'ABB DC Breaker',
    description: 'Miniature circuit breaker with model and ratings',
    icon: <ScanSearch className="size-4" />,
    imagePath: '/demo-abb-breaker.jpg',
  },
]

interface CameraStageProps {
  state: StageState
  selectedImage: SelectedImage | null
  onImageSelected: (selection: SelectedImage) => Promise<void>
  onReset: () => void
}

export function CameraStage({
  state,
  selectedImage,
  onImageSelected,
  onReset,
}: CameraStageProps) {
  const [isSelecting, setIsSelecting] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const openFilePicker = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    event.target.value = ''
    if (!file) return

    setIsSelecting(true)
    try {
      await onImageSelected({
        id: `upload-${Date.now()}`,
        name: file.name,
        previewUrl: URL.createObjectURL(file),
        file,
      })
    } finally {
      setIsSelecting(false)
    }
  }

  const handleDemoClick = async (demo: DemoComponent) => {
    setIsSelecting(true)
    try {
      const response = await fetch(demo.imagePath)
      if (!response.ok) {
        throw new Error(`Failed to load demo image: ${demo.name}`)
      }
      const blob = await response.blob()
      const file = new File([blob], `${demo.id}.jpg`, {
        type: blob.type || 'image/jpeg',
      })
      await onImageSelected({
        id: demo.id,
        name: demo.name,
        previewUrl: demo.imagePath,
        file,
      })
    } finally {
      setIsSelecting(false)
    }
  }

  return (
    <div className="relative flex min-h-[44vh] min-w-0 flex-col overflow-hidden border-b border-slate-300/70 bg-[linear-gradient(180deg,#0f172a_0%,#111827_58%,#0b1220_100%)] sm:min-h-[50vh] xl:min-h-0 xl:h-full xl:border-b-0">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        className="hidden"
        onChange={handleFileChange}
      />

      {state === 'default' && (
        <DefaultState
          isSelecting={isSelecting}
          onStartCamera={openFilePicker}
          onDemoSelect={handleDemoClick}
        />
      )}

      {(state === 'active' || state === 'scanning') && selectedImage && (
        <ActiveState
          image={selectedImage}
          isScanning={state === 'scanning'}
          onUploadNewImage={openFilePicker}
          onReset={onReset}
        />
      )}
    </div>
  )
}

function DefaultState({
  isSelecting,
  onStartCamera,
  onDemoSelect,
}: {
  isSelecting: boolean
  onStartCamera: () => void
  onDemoSelect: (demo: DemoComponent) => Promise<void>
}) {
  return (
    <div className="flex min-h-0 flex-1 flex-col overflow-auto px-5 py-6 sm:px-8 sm:py-8">
      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.15fr)_minmax(18rem,24rem)]">
        <section className="rounded-[2rem] border border-white/10 bg-white/6 p-6 text-white shadow-[0_24px_80px_-46px_rgba(15,23,42,0.9)] backdrop-blur-md sm:p-8">
          <div className="flex max-w-2xl flex-col gap-6">
            <div className="inline-flex w-fit items-center gap-2 rounded-full border border-sky-400/30 bg-sky-500/10 px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-100">
              <ScanSearch className="size-3.5" />
              Camera-first diagnostic workflow
            </div>
            <div>
              <h2 className="text-3xl font-semibold tracking-tight text-white sm:text-4xl">
                Capture a component and move straight to cited technical answers.
              </h2>
              <p className="mt-4 max-w-2xl text-sm leading-7 text-slate-300 sm:text-base">
                The assistant identifies the component, extracts part data from
                the label, prepares official documentation in the background,
                and keeps the chat grounded in cited manufacturer sources.
              </p>
            </div>
            <div className="grid gap-3 text-sm text-slate-200 sm:grid-cols-3">
              <WorkflowCard
                step="01"
                title="Capture"
                description="Upload a label shot or use a curated industrial demo sample."
              />
              <WorkflowCard
                step="02"
                title="Identify"
                description="Gemini extracts manufacturer, model, and usable part references."
              />
              <WorkflowCard
                step="03"
                title="Inspect"
                description="Ask for ratings, standards, or manuals and get cited answers."
              />
            </div>
            <div className="flex flex-wrap gap-3">
              <Button
                onClick={onStartCamera}
                className="h-11 rounded-full px-5 text-sm shadow-[0_16px_40px_-22px_rgba(37,99,235,0.9)]"
                disabled={isSelecting}
              >
                <Camera className="size-4" />
                Capture or upload image
              </Button>
              <span className="inline-flex items-center rounded-full border border-white/12 bg-white/6 px-4 py-2 text-xs uppercase tracking-[0.18em] text-slate-300">
                Chroma mode active
              </span>
            </div>
          </div>
        </section>

        <aside className="rounded-[2rem] border border-slate-200/80 bg-white/92 p-5 shadow-[0_24px_70px_-40px_rgba(15,23,42,0.45)]">
          <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-slate-500">
            Demo components
          </p>
          <div className="mt-4 grid gap-3">
            {demoComponents.map((demo) => (
              <button
                key={demo.id}
                onClick={() => void onDemoSelect(demo)}
                className="group flex items-center gap-3 rounded-2xl border border-slate-200 bg-slate-50/70 p-3 text-left transition-all hover:-translate-y-0.5 hover:border-sky-300 hover:bg-white hover:shadow-md active:scale-[0.99]"
                type="button"
                disabled={isSelecting}
              >
                <div className="relative size-16 shrink-0 overflow-hidden rounded-xl border border-slate-200 bg-white">
                  <img
                    src={demo.imagePath}
                    alt={demo.name}
                    className="size-full object-cover"
                  />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-2">
                    <span className="inline-flex size-7 items-center justify-center rounded-full bg-sky-50 text-sky-700">
                      {demo.icon}
                    </span>
                    <p className="truncate text-sm font-semibold text-slate-900">
                      {demo.name}
                    </p>
                  </div>
                  <p className="mt-1 text-xs leading-5 text-slate-500">
                    {demo.description}
                  </p>
                </div>
                <ChevronRight className="size-4 shrink-0 text-slate-400 transition-transform group-hover:translate-x-0.5" />
              </button>
            ))}
          </div>
        </aside>
      </div>
    </div>
  )
}

function WorkflowCard({
  step,
  title,
  description,
}: {
  step: string
  title: string
  description: string
}) {
  return (
    <div className="rounded-2xl border border-white/10 bg-white/6 p-4">
      <p className="text-[11px] font-semibold uppercase tracking-[0.2em] text-sky-200">
        {step}
      </p>
      <p className="mt-2 text-sm font-semibold text-white">{title}</p>
      <p className="mt-1 text-sm leading-6 text-slate-300">{description}</p>
    </div>
  )
}

function ActiveState({
  image,
  isScanning,
  onUploadNewImage,
  onReset,
}: {
  image: SelectedImage
  isScanning: boolean
  onUploadNewImage: () => void
  onReset: () => void
}) {
  return (
    <div className="relative flex min-h-0 flex-1 items-center justify-center overflow-hidden p-4 sm:p-6 xl:p-8">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top,rgba(56,189,248,0.08),transparent_28%),linear-gradient(180deg,rgba(15,23,42,0.18),rgba(15,23,42,0.46))]" />

      <div className="absolute left-4 top-4 z-10 flex max-w-[calc(100%-2rem)] flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={onReset}
          className="glass rounded-full border-white/20 bg-white/10 px-4 text-white hover:bg-white/20 hover:text-white"
        >
          <ArrowLeft className="size-4" />
          Back
        </Button>
        <div className="glass-dark max-w-[16rem] rounded-full border border-white/10 px-4 py-2 text-sm text-slate-100 shadow-lg">
          <span className="block truncate">{image.name}</span>
        </div>
      </div>

      <div className="absolute bottom-4 left-4 z-10 flex max-w-[calc(100%-2rem)] flex-wrap items-center gap-2">
        <div className="glass-dark rounded-full border border-white/10 px-4 py-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-100">
          Still image analysis
        </div>
        <div className="glass-dark rounded-full border border-white/10 px-4 py-2 text-xs font-medium uppercase tracking-[0.18em] text-slate-300">
          Label framing enabled
        </div>
      </div>

      <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-2">
        <Button
          onClick={onUploadNewImage}
          type="button"
          className="rounded-full bg-white/92 px-4 text-slate-900 shadow-[0_18px_40px_-24px_rgba(15,23,42,0.95)] hover:bg-white"
        >
          <Upload className="size-4" />
          Replace image
        </Button>
      </div>

      <div className="relative flex h-full w-full max-w-[1200px] items-center justify-center overflow-hidden rounded-[2rem] border border-white/10 bg-slate-950/82 shadow-[0_36px_90px_-52px_rgba(0,0,0,0.95)]">
        <img
          src={image.previewUrl}
          alt={image.name}
          className="absolute inset-0 size-full bg-slate-950 object-contain p-4 sm:p-6 xl:p-8"
        />

        <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
          <div className="relative h-[min(60vh,34rem)] w-[min(88vw,44rem)] max-w-[calc(100%-2rem)]">
            <div className="absolute left-0 top-0 size-9 border-l-2 border-t-2 border-white/80" />
            <div className="absolute right-0 top-0 size-9 border-r-2 border-t-2 border-white/80" />
            <div className="absolute bottom-0 left-0 size-9 border-b-2 border-l-2 border-white/80" />
            <div className="absolute bottom-0 right-0 size-9 border-b-2 border-r-2 border-white/80" />

            {isScanning && (
              <div className="absolute inset-x-0 top-0 h-full overflow-hidden">
                <div className="animate-scan h-0.5 w-full bg-gradient-to-r from-transparent via-sky-300 to-transparent" />
              </div>
            )}
          </div>
        </div>

        {isScanning && (
          <div className="absolute left-1/2 top-6 z-10 -translate-x-1/2">
            <div className="glass rounded-full border border-slate-200/70 px-4 py-2 shadow-lg">
              <div className="flex items-center gap-2 text-sm font-medium text-slate-900">
                <div className="size-2 animate-pulse rounded-full bg-primary" />
                Analyzing component and preparing documentation...
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
