'use client'

import { Clock, Cpu } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetTrigger,
} from '@/components/ui/sheet'
import { ScrollArea } from '@/components/ui/scroll-area'

interface HistoryItem {
  id: string
  manufacturer: string
  model: string
  timestamp: string
  thumbnail: string
}

const mockHistory: HistoryItem[] = [
  {
    id: '1',
    manufacturer: 'Siemens',
    model: 'S7-1200',
    timestamp: 'Today, 14:30',
    thumbnail: '/api/placeholder/60/60',
  },
  {
    id: '2',
    manufacturer: 'Allen-Bradley',
    model: 'CompactLogix 5380',
    timestamp: 'Today, 11:15',
    thumbnail: '/api/placeholder/60/60',
  },
  {
    id: '3',
    manufacturer: 'Schneider Electric',
    model: 'Modicon M340',
    timestamp: 'Yesterday, 16:45',
    thumbnail: '/api/placeholder/60/60',
  },
  {
    id: '4',
    manufacturer: 'ASUS',
    model: 'ROG Maximus Z790',
    timestamp: 'Yesterday, 09:20',
    thumbnail: '/api/placeholder/60/60',
  },
]

export function Header() {
  return (
    <header className="sticky top-0 z-40 flex h-14 items-center justify-between border-b bg-card px-4 shadow-sm">
      <div className="flex items-center gap-2">
        <div className="flex size-8 items-center justify-center rounded-lg bg-primary">
          <Cpu className="size-5 text-primary-foreground" />
        </div>
        <span className="text-lg font-semibold text-foreground">
          Visual Technical Assistant
        </span>
      </div>

      <Sheet>
        <SheetTrigger asChild>
          <Button
            variant="ghost"
            size="icon"
            className="transition-transform active:scale-95"
            aria-label="View scan history"
          >
            <Clock className="size-5" />
          </Button>
        </SheetTrigger>
        <SheetContent className="w-80 sm:w-96">
          <SheetHeader>
            <SheetTitle>Scan History</SheetTitle>
          </SheetHeader>
          <ScrollArea className="mt-4 h-[calc(100vh-8rem)]">
            <div className="flex flex-col gap-3 pr-4">
              {mockHistory.map((item) => (
                <HistoryCard key={item.id} item={item} />
              ))}
            </div>
          </ScrollArea>
        </SheetContent>
      </Sheet>
    </header>
  )
}

function HistoryCard({ item }: { item: HistoryItem }) {
  return (
    <button
      className="flex w-full items-center gap-3 rounded-xl border bg-card p-3 text-left shadow-sm transition-all hover:border-primary/30 hover:shadow-md active:scale-[0.98]"
      type="button"
    >
      <div className="size-12 shrink-0 overflow-hidden rounded-lg bg-muted">
        <div className="flex size-full items-center justify-center text-muted-foreground">
          <Cpu className="size-5" />
        </div>
      </div>
      <div className="min-w-0 flex-1">
        <p className="truncate text-sm font-medium text-foreground">
          {item.manufacturer} {item.model}
        </p>
        <p className="text-xs text-muted-foreground">{item.timestamp}</p>
      </div>
    </button>
  )
}
