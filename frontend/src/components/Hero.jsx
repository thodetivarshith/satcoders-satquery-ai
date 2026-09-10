const STATS = [
  { value: "2", label: "Sensing modes fused", detail: "Optical + SAR" },
  { value: "75%+", label: "Target VQA accuracy", detail: "GeoChat / BigEarthNet" },
  { value: "<3s", label: "Answer latency", detail: "Upload to evidence" },
];

/**
 * The one orchestrated load moment for the whole app: headline, subhead
 * and stat row reveal in a single staggered sequence, then everything
 * else is quiet. Grounded in the product's actual spec (fusion, GeoChat,
 * agentic routing) rather than generic SaaS hero copy.
 */
function Hero() {
  return (
    <section className="panel panel-lg relative overflow-hidden px-5 py-10 sm:px-10 sm:py-14">
      {/* Ambient orbit rings — evokes a sensing pass without being literal */}
      <div
        className="pointer-events-none absolute -right-24 -top-24 h-[420px] w-[420px] opacity-[0.35] sm:-right-16 sm:-top-32"
        aria-hidden="true"
      >
        <svg viewBox="0 0 400 400" className="h-full w-full">
          <circle cx="200" cy="200" r="180" fill="none" stroke="var(--optical)" strokeOpacity="0.18" strokeWidth="1" />
          <circle cx="200" cy="200" r="140" fill="none" stroke="var(--optical)" strokeOpacity="0.14" strokeWidth="1" />
          <g className="orbit-ring-a" style={{ transformOrigin: "200px 200px" }}>
            <circle cx="200" cy="200" r="180" fill="none" stroke="var(--optical)" strokeOpacity="0.5" strokeWidth="1.5" strokeDasharray="2 14" />
          </g>
          <g className="orbit-ring-b" style={{ transformOrigin: "200px 200px" }}>
            <circle cx="200" cy="200" r="100" fill="none" stroke="var(--sar)" strokeOpacity="0.4" strokeWidth="1.5" strokeDasharray="1 10" />
          </g>
        </svg>
      </div>
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse 700px 420px at 20% 20%, rgba(94,200,216,0.1), transparent 65%)",
        }}
        aria-hidden="true"
      />

      <div className="relative max-w-2xl">
        <p className="rise-in text-xs font-medium text-text-faint" style={{ fontFamily: "var(--font-mono)" }}>
          SIH26167 · Remote sensing analysis
        </p>
        <h1
          className="rise-in mt-3 text-3xl font-semibold leading-[1.08] tracking-tight sm:text-[2.75rem]"
          style={{ fontFamily: "var(--font-display)", animationDelay: "70ms" }}
        >
          <span className="text-gradient">Ask a satellite image</span>
          <br />
          <span className="text-text">anything.</span>
        </h1>
        <p
          className="rise-in mt-4 max-w-md text-sm leading-relaxed text-text-muted sm:text-base"
          style={{ animationDelay: "140ms" }}
        >
          Upload optical or SAR imagery, ask a plain-language question, and a routed
          GeoChat pipeline grounds its answer in the pixels — with evidence, not just a guess.
        </p>

        <dl
          className="rise-in mt-8 grid grid-cols-3 gap-3 sm:gap-6"
          style={{ animationDelay: "210ms" }}
        >
          {STATS.map((stat, index) => (
            <div key={stat.label} className="count-blip" style={{ animationDelay: `${260 + index * 90}ms` }}>
              <dt className="sr-only">{stat.label}</dt>
              <dd
                className="text-xl font-semibold tracking-tight text-text sm:text-2xl"
                style={{ fontFamily: "var(--font-display)" }}
              >
                {stat.value}
              </dd>
              <p className="mt-0.5 text-[11px] leading-snug text-text-muted sm:text-xs">{stat.label}</p>
              <p className="text-[10px] text-text-faint" style={{ fontFamily: "var(--font-mono)" }}>
                {stat.detail}
              </p>
            </div>
          ))}
        </dl>
      </div>
    </section>
  );
}

export default Hero;
