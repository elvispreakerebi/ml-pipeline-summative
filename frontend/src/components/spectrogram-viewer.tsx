"use client";

interface SpectrogramViewerProps {
  base64Image: string;
  alt?: string;
}

export function SpectrogramViewer({
  base64Image,
  alt = "Mel Spectrogram",
}: SpectrogramViewerProps) {
  return (
    <div className="overflow-hidden rounded-lg border">
      <img
        src={`data:image/png;base64,${base64Image}`}
        alt={alt}
        className="w-full"
      />
    </div>
  );
}
