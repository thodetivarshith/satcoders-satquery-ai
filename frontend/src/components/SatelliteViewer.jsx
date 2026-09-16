import { useCallback, useRef, useState } from "react";
import BoundingBoxOverlay from "./BoundingBoxOverlay";
import ImageMetadataPanel from "./ImageMetadataPanel";

const ACCEPTED_TYPES = ["image/png", "image/jpeg"];
const ACCEPTED_EXTENSIONS = /\.(png|jpe?g|tif|tiff)$/i;

/**
 * The main analysis viewport: upload target, satellite image preview,
 * zoom/pan/reset/fullscreen controls, and — once a result exists — the
 * interactive bounding-box evidence overlay.
 */
function SatelliteViewer({
  file,
  imageUrl,
  dimensions,
  onImageSelect,
  onClear,
  onDimensionsResolved,
  result,
  selectedBoxIndex,
  onSelectBox,
  previewUnavailable,
  onPreviewError,
  isAnalyzing,
}) {
  const [isDragging, setIsDragging] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [uploadWarning, setUploadWarning] = useState("");
  const [cursorPos, setCursorPos] = useState(null);
  const dragState = useRef(null);
  const inputRef = useRef(null);
  const viewportRef = useRef(null);
  const [isFullscreen, setIsFullscreen] = useState(false);

  const validateAndSelect = useCallback(
    (candidate) => {
      if (!candidate) return;
      const looksAccepted =
        ACCEPTED_TYPES.includes(candidate.type) ||
        ACCEPTED_EXTENSIONS.test(candidate.name || "");
      if (!looksAccepted) {
        setUploadWarning("This image format could not be processed.");
        return;
      }
      setUploadWarning("");
      setZoom(1);
      setPan({ x: 0, y: 0 });
      onImageSelect(candidate);
    },
    [onImageSelect]
  );

  const handleFileChange = (e) => validateAndSelect(e.target.files[0]);

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    validateAndSelect(e.dataTransfer.files[0]);
  };

  const handleImgLoad = (e) => {
    onDimensionsResolved?.({
      width: e.target.naturalWidth,
      height: e.target.naturalHeight,
    });
  };

  const handlePointerDown = (e) => {
    if (zoom <= 1) return;
    dragState.current = { startX: e.clientX, startY: e.clientY, origin: pan };
  };
  const handlePointerMove = (e) => {
    if (dragState.current) {
      const dx = e.clientX - dragState.current.startX;
      const dy = e.clientY - dragState.current.startY;
      setPan({ x: dragState.current.origin.x + dx, y: dragState.current.origin.y + dy });
    }
    const rect = viewportRef.current?.getBoundingClientRect();
    if (rect) {
      const xPct = ((e.clientX - rect.left) / rect.width) * 100;
      const yPct = ((e.clientY - rect.top) / rect.height) * 100;
      if (xPct >= 0 && xPct <= 100 && yPct >= 0 && yPct <= 100) {
        setCursorPos({ x: xPct, y: yPct });
      }
    }
  };
  const handlePointerUp = () => {
    dragState.current = null;
  };
  const handlePointerLeaveViewport = () => {
    dragState.current = null;
    setCursorPos(null);
  };

  const zoomIn = () => setZoom((z) => Math.min(z + 0.25, 3));
  const zoomOut = () =>
    setZoom((z) => {
      const next = Math.max(z - 0.25, 1);
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  const resetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const toggleFullscreen = () => {
    if (!viewportRef.current) return;
    if (!document.fullscreenElement) {
      viewportRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  return (
    <section className="panel-frame" aria-label="Satellite image workspace">
      <div className="panel-frame-inner">
      <div className="flex items-center justify-between border-b border-border-hair px-4 py-3 sm:px-5">
        <h2 className="panel-title">Satellite image</h2>
        {file && (
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1.5 rounded-full border border-border px-3 py-1 text-xs text-text-muted transition hover:border-optical/50 hover:text-text"
          >
            <span aria-hidden="true">←</span> Back to upload
          </button>
        )}
      </div>

      {!file ? (
        <label
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          className={`flex min-h-[22rem] cursor-pointer flex-col items-center justify-center gap-3 border-2 border-dashed p-8 text-center transition ${
            isDragging ? "border-optical bg-[rgba(94,200,216,0.08)] glow-optical" : "border-border-strong"
          }`}
        >
          <div
            className="flex h-12 w-12 items-center justify-center rounded-full border border-border-strong text-text-muted"
            aria-hidden="true"
          >
            ↑
          </div>
          <p className="text-sm font-medium text-text">
            Drag a satellite image here
          </p>
          <p className="text-xs text-text-muted">or click to browse · PNG, JPG, GeoTIFF</p>
          {uploadWarning && (
            <p className="text-xs text-danger" role="alert">
              {uploadWarning}
            </p>
          )}
          <input
            ref={inputRef}
            type="file"
            accept="image/png,image/jpeg,.tif,.tiff"
            onChange={handleFileChange}
            className="hidden"
            aria-label="Upload satellite image"
          />
        </label>
      ) : (
        <>
          <div
            ref={viewportRef}
            className="relative h-[22rem] overflow-hidden bg-black sm:h-[30rem]"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeaveViewport}
            style={{ cursor: zoom > 1 ? "grab" : "default" }}
          >
            {previewUnavailable ? (
              <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
                <p className="text-sm text-text-muted">
                  Preview isn't available for this format in-browser.
                </p>
                <p className="text-xs text-text-faint">
                  {file.name} will still be sent for analysis.
                </p>
              </div>
            ) : (
              <>
                <img
                  src={imageUrl}
                  onLoad={handleImgLoad}
                  onError={() => onPreviewError?.()}
                  alt="Selected satellite scene"
                  draggable={false}
                  className="absolute left-1/2 top-1/2 max-h-none max-w-none select-none"
                  style={{
                    transform: `translate(-50%, -50%) translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                    transition: dragState.current ? "none" : "transform 150ms ease-out",
                    height: "100%",
                    width: "100%",
                    objectFit: "contain",
                  }}
                />
                <BoundingBoxOverlay
                  boxes={result?.bounding_boxes}
                  selectedIndex={selectedBoxIndex}
                  onSelect={onSelectBox}
                />
              </>
            )}

            {isAnalyzing && !previewUnavailable && (
              <span
                className="scan-line pointer-events-none absolute left-0 right-0 h-px"
                style={{ background: "linear-gradient(90deg, transparent, var(--optical), transparent)" }}
                aria-hidden="true"
              />
            )}

            {cursorPos && !previewUnavailable && zoom === 1 && !dragState.current && (
              <>
                <span
                  className="pointer-events-none absolute h-full w-px opacity-30"
                  style={{ left: `${cursorPos.x}%`, background: "var(--optical)" }}
                  aria-hidden="true"
                />
                <span
                  className="pointer-events-none absolute h-px w-full opacity-30"
                  style={{ top: `${cursorPos.y}%`, background: "var(--optical)" }}
                  aria-hidden="true"
                />
                <span
                  className="pointer-events-none absolute rounded bg-[rgba(10,12,16,0.8)] px-1.5 py-0.5 text-[9px] text-optical"
                  style={{
                    left: `${cursorPos.x}%`,
                    top: `${cursorPos.y}%`,
                    transform: "translate(10px, 10px)",
                    fontFamily: "var(--font-mono)",
                  }}
                >
                  {cursorPos.x.toFixed(0)}, {cursorPos.y.toFixed(0)}
                </span>
              </>
            )}

            {/* Vignette — subtle depth cue at the frame edges, not a decorative wash */}
            <div
              className="pointer-events-none absolute inset-0"
              style={{
                boxShadow: "inset 0 0 60px 10px rgba(0,0,0,0.35)",
              }}
              aria-hidden="true"
            />

            {/* Reticle frame — the viewport reads as an analysis instrument, not a photo frame */}
            <Corner className="left-3 top-3 border-l border-t" />
            <Corner className="right-3 top-3 border-r border-t" />
            <Corner className="bottom-3 left-3 border-b border-l" />
            <Corner className="bottom-3 right-3 border-b border-r" />

            {!previewUnavailable && (
              <button
                type="button"
                onClick={onClear}
                aria-label="Back to upload"
                className="absolute left-3 top-3 flex items-center gap-1.5 rounded-full border border-border-strong bg-[rgba(10,12,16,0.85)] px-2.5 py-1.5 text-[11px] font-medium text-text-muted backdrop-blur-md transition hover:border-optical/60 hover:text-optical"
              >
                <span aria-hidden="true">←</span> Back
              </button>
            )}

            {dimensions && (
              <div
                className="absolute left-3 bottom-3 rounded bg-[rgba(10,12,16,0.75)] px-2 py-1 text-[10px] text-text-muted"
                style={{ fontFamily: "var(--font-mono)" }}
              >
                {dimensions.width}×{dimensions.height}px · {Math.round(zoom * 100)}%
              </div>
            )}

            <div className="absolute right-3 bottom-3 flex overflow-hidden rounded-lg border border-border-strong bg-[rgba(10,12,16,0.85)] backdrop-blur-md">
              <ViewportButton onClick={zoomOut} label="Zoom out">
                −
              </ViewportButton>
              <ViewportButton onClick={zoomIn} label="Zoom in">
                +
              </ViewportButton>
              <ViewportButton onClick={resetView} label="Reset view">
                ⟲
              </ViewportButton>
              <ViewportButton onClick={toggleFullscreen} label="Toggle fullscreen">
                {isFullscreen ? "⤡" : "⤢"}
              </ViewportButton>
            </div>
          </div>

          <ImageMetadataPanel
            file={file}
            dimensions={dimensions}
            backendMetadata={result?.metadata}
          />
        </>
      )}
      </div>
    </section>
  );
}

function Corner({ className }) {
  return (
    <span
      className={`pointer-events-none absolute h-4 w-4 border-border-strong ${className}`}
      aria-hidden="true"
    />
  );
}

function ViewportButton({ onClick, label, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      className="flex h-8 w-8 items-center justify-center text-sm text-text-muted transition hover:bg-surface-raised hover:text-optical"
    >
      {children}
    </button>
  );
}

export default SatelliteViewer;
