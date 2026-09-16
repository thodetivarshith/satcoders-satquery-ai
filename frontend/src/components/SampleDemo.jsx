import { SAMPLE_SCENES } from "../data/samples";

/**
 * Lets a judge see the full pipeline without sourcing their own image.
 * Visual thumbnail cards (not plain text pills) so the sample strip
 * itself reads as part of the analysis product, not a settings row.
 * Loading a sample is explicit and clearly labeled — it never pretends
 * to be a live backend call (see the DEMO DATA badge in ResultPanel).
 */
function SampleDemo({ onSelectSample, disabled, activeSampleId }) {
  return (
    <section className="panel p-4 sm:p-5">
      <div className="flex items-center justify-between">
        <h2 className="panel-title">Try a sample scene</h2>
        <span className="text-[10px] uppercase tracking-wide text-text-faint">
          No upload needed
        </span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-2 sm:grid-cols-5">
        {SAMPLE_SCENES.map((sample, index) => {
          const isActive = activeSampleId === sample.id;
          return (
            <button
              key={sample.id}
              type="button"
              onClick={() => onSelectSample(sample)}
              disabled={disabled}
              className={`rise-in group relative overflow-hidden rounded-lg border text-left transition disabled:cursor-not-allowed disabled:opacity-40 ${
                isActive
                  ? "border-optical glow-optical"
                  : "border-border-hair bg-bg-elevated/40 hover:border-border-strong hover:-translate-y-0.5"
              }`}
              style={{ animationDelay: `${index * 60}ms` }}
            >
              <div className="relative h-14 w-full overflow-hidden bg-bg sm:h-16">
                <img
                  src={sample.image}
                  alt=""
                  aria-hidden="true"
                  className="h-full w-full object-cover transition duration-300 group-hover:scale-105"
                  draggable={false}
                />
                <div
                  className="absolute inset-0"
                  style={{
                    background: isActive
                      ? "linear-gradient(180deg, transparent 40%, rgba(94,200,216,0.25) 100%)"
                      : "linear-gradient(180deg, transparent 50%, rgba(10,12,16,0.65) 100%)",
                  }}
                />
                {isActive && (
                  <span
                    className="absolute right-1 top-1 flex h-4 w-4 items-center justify-center rounded-full bg-optical text-[9px] font-bold text-bg"
                    aria-hidden="true"
                  >
                    ✓
                  </span>
                )}
              </div>
              <div className="flex items-center gap-1 px-1.5 py-1.5">
                <span className="text-[11px] leading-none" aria-hidden="true">
                  {sample.icon}
                </span>
                <span className="truncate text-[10px] font-medium text-text">
                  {sample.label}
                </span>
              </div>
            </button>
          );
        })}
      </div>
    </section>
  );
}

export default SampleDemo;
