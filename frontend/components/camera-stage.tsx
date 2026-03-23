'use client'

import { useRef, useState, type ChangeEvent, type ReactNode } from 'react'
import {
  Camera,
  Flashlight,
  FlipHorizontal,
  Router,
  Cpu,
  MousePointer2,
  Upload,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { ScrollArea, ScrollBar } from '@/components/ui/scroll-area'

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
  icon: ReactNode
  imagePath: string
}

const demoComponents: DemoComponent[] = [
  {
    id: 'huawei-router',
    name: 'Huawei Router',
    icon: <Router className="size-5" />,
    imagePath: '/demo-router.jpg',
  },
  {
    id: 'ajazz-receiver',
    name: 'AJAZZ Receiver',
    icon: <MousePointer2 className="size-5" />,
    imagePath: '/demo-receiver.jpg',
  },
]

interface CameraStageProps {
  state: StageState
  selectedImage: SelectedImage | null
  onImageSelected: (selection: SelectedImage) => Promise<void>
}

export function CameraStage({
  state,
  selectedImage,
  onImageSelected,
}: CameraStageProps) {
  const [flashOn, setFlashOn] = useState(false)
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
    <div className="relative flex h-[45vh] flex-col bg-muted lg:h-full lg:flex-1">
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
          flashOn={flashOn}
          onFlashToggle={() => setFlashOn(!flashOn)}
          onUploadNewImage={openFilePicker}
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
    <div className="flex flex-1 flex-col items-center justify-center gap-6 p-6">
      <div className="flex flex-col items-center gap-4 text-center">
        <div className="flex size-16 items-center justify-center rounded-full bg-primary/10">
          <Camera className="size-8 text-primary" />
        </div>
        <div>
          <h2 className="text-lg font-semibold text-foreground">
            Identify a Component
          </h2>
          <p className="mt-1 text-sm text-muted-foreground">
            Upload a photo, capture one from your device, or try a real demo
            image from this project
          </p>
        </div>
        <div className="flex flex-col gap-2 sm:flex-row">
          <Button
            onClick={onStartCamera}
            className="gap-2 transition-transform active:scale-95"
            disabled={isSelecting}
          >
            <Camera className="size-4" />
            Capture or Upload
          </Button>
        </div>
      </div>

      <div className="w-full max-w-md">
        <p className="mb-3 text-center text-xs font-medium uppercase tracking-wider text-muted-foreground">
          Project demos
        </p>
        <ScrollArea className="w-full whitespace-nowrap">
          <div className="flex gap-3 pb-2">
            {demoComponents.map((demo) => (
              <button
                key={demo.id}
                onClick={() => void onDemoSelect(demo)}
                className="group flex flex-col items-center gap-2 transition-transform active:scale-95"
                type="button"
                disabled={isSelecting}
              >
                <div className="relative size-20 overflow-hidden rounded-2xl border-2 border-transparent bg-card shadow-sm transition-all group-hover:border-primary/50 group-hover:shadow-md">
                  <img
                    src={demo.imagePath}
                    alt={demo.name}
                    className="size-full object-cover"
                  />
                  <div className="absolute inset-0 flex items-center justify-center bg-black/25 opacity-0 transition-opacity group-hover:opacity-100">
                    {demo.icon}
                  </div>
                </div>
                <span className="text-xs font-medium text-muted-foreground group-hover:text-foreground">
                  {demo.name}
                </span>
              </button>
            ))}
          </div>
          <ScrollBar orientation="horizontal" />
        </ScrollArea>
      </div>
    </div>
  )
}

function ActiveState({
  image,
  isScanning,
  flashOn,
  onFlashToggle,
  onUploadNewImage,
}: {
  image: SelectedImage
  isScanning: boolean
  flashOn: boolean
  onFlashToggle: () => void
  onUploadNewImage: () => void
}) {
  return (
    <div className="relative flex-1">
      <img
        src={image.previewUrl}
        alt={image.name}
        className="absolute inset-0 size-full object-cover"
      />

      <div className="pointer-events-none absolute inset-0 flex items-center justify-center">
        <div className="relative size-48 sm:size-64">
          <div className="absolute left-0 top-0 size-8 border-l-2 border-t-2 border-white/80" />
          <div className="absolute right-0 top-0 size-8 border-r-2 border-t-2 border-white/80" />
          <div className="absolute bottom-0 left-0 size-8 border-b-2 border-l-2 border-white/80" />
          <div className="absolute bottom-0 right-0 size-8 border-b-2 border-r-2 border-white/80" />

          {isScanning && (
            <div className="absolute inset-x-0 top-0 h-full overflow-hidden">
              <div className="animate-scan h-0.5 w-full bg-gradient-to-r from-transparent via-primary to-transparent" />
            </div>
          )}
        </div>
      </div>

      {isScanning && (
        <div className="absolute left-1/2 top-6 -translate-x-1/2">
          <div className="glass flex items-center gap-2 rounded-full px-4 py-2 shadow-lg">
            <div className="size-2 animate-pulse rounded-full bg-primary" />
            <span className="text-sm font-medium text-foreground">
              Analyzing component...
            </span>
          </div>
        </div>
      )}

      <div className="absolute left-4 top-4">
        <div className="glass rounded-full px-3 py-1.5 text-sm font-medium text-foreground shadow-lg">
          {image.name}
        </div>
      </div>

      <div className="absolute bottom-6 right-4 flex flex-col gap-2">
        <button
          onClick={onUploadNewImage}
          className="glass flex size-12 items-center justify-center rounded-full shadow-lg transition-all active:scale-95"
          type="button"
          aria-label="Upload a new image"
        >
          <Upload className="size-5" />
        </button>
        <button
          onClick={onFlashToggle}
          className={`glass flex size-12 items-center justify-center rounded-full shadow-lg transition-all active:scale-95 ${
            flashOn ? 'bg-primary text-primary-foreground' : ''
          }`}
          type="button"
          aria-label="Toggle flashlight"
        >
          <Flashlight className="size-5" />
        </button>
        <button
          className="glass flex size-12 items-center justify-center rounded-full shadow-lg transition-all active:scale-95"
          type="button"
          aria-label="Switch camera"
        >
          <FlipHorizontal className="size-5" />
        </button>
      </div>
    </div>
  )
}
