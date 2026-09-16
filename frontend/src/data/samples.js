// Sample demo scenes for judges/users who don't want to source
// their own satellite image mid-demo.
//
// IMPORTANT: these are procedurally generated placeholder graphics,
// not real satellite imagery — swap `image` for real sample scenes
// (e.g. from Kunchala's preprocessing test set) before the live demo.
// The analysis result returned for these is explicitly mocked and
// labeled as DEMO DATA in the UI; it is never presented as a live
// backend response.

function svgDataUri(svgMarkup) {
  return `data:image/svg+xml;utf8,${encodeURIComponent(svgMarkup)}`;
}

function scene({ bg, blocks }) {
  const rects = blocks
    .map(
      (b) =>
        `<rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" fill="${b.fill}" opacity="${b.opacity ?? 1}" />`
    )
    .join("");
  return svgDataUri(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="480" viewBox="0 0 640 480">
      <rect width="640" height="480" fill="${bg}" />
      ${rects}
    </svg>`
  );
}

export const SAMPLE_SCENES = [
  {
    id: "agriculture",
    label: "Agriculture",
    icon: "\u25A6",
    query: "What type of land cover is visible?",
    description: "Cultivated field parcels with visible crop rows.",
    image: scene({
      bg: "#2b3a1f",
      blocks: [
        { x: 0, y: 0, w: 640, h: 480, fill: "#3d5528" },
        { x: 0, y: 0, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
        { x: 0, y: 80, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
        { x: 0, y: 160, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
        { x: 0, y: 240, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
        { x: 0, y: 320, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
        { x: 0, y: 400, w: 640, h: 40, fill: "#4a6631", opacity: 0.6 },
      ],
    }),
    mockResult: {
      query: "What type of land cover is visible?",
      answer:
        "The scene shows predominantly agricultural land, organized into parallel cultivated parcels consistent with row-crop farming. No significant built-up structures are visible.",
      confidence: 0.91,
      task_type: "vqa",
      execution_trace: {
        router_decision: "vqa",
        models_called: ["geochat"],
        processing_time_ms: 980,
      },
      bounding_boxes: [
        { label: "cultivated field", confidence: 0.89, x_min: 0.05, y_min: 0.1, x_max: 0.95, y_max: 0.9 },
      ],
      metadata: { demo: true, source: "synthetic sample" },
    },
  },
  {
    id: "water",
    label: "Water Body",
    icon: "\u2248",
    query: "Where are the water bodies?",
    description: "Coastal or riverine scene with a visible water boundary.",
    image: scene({
      bg: "#173040",
      blocks: [
        { x: 0, y: 0, w: 640, h: 260, fill: "#264a34" },
        { x: 0, y: 260, w: 640, h: 220, fill: "#1c5b78" },
      ],
    }),
    mockResult: {
      query: "Where are the water bodies?",
      answer:
        "A large water body occupies the lower portion of the image, with a fairly regular shoreline separating it from vegetated land to the north.",
      confidence: 0.94,
      task_type: "grounding",
      execution_trace: {
        router_decision: "grounding",
        models_called: ["router", "grounding"],
        processing_time_ms: 1120,
      },
      bounding_boxes: [
        { label: "water body", confidence: 0.94, x_min: 0.0, y_min: 0.54, x_max: 1.0, y_max: 1.0 },
      ],
      metadata: { demo: true, source: "synthetic sample" },
    },
  },
  {
    id: "urban",
    label: "Urban Area",
    icon: "\u25A3",
    query: "Identify buildings.",
    description: "Dense grid of built-up structures and roads.",
    image: scene({
      bg: "#23262e",
      blocks: Array.from({ length: 24 }, (_, i) => ({
        x: (i % 6) * 106 + 10,
        y: Math.floor(i / 6) * 116 + 10,
        w: 80,
        h: 90,
        fill: i % 3 === 0 ? "#3a3f4c" : "#4a5063",
      })),
    }),
    mockResult: {
      query: "Identify buildings.",
      answer:
        "The image shows a dense urban grid with regularly spaced rectangular structures consistent with residential or light-industrial buildings, separated by a visible road network.",
      confidence: 0.88,
      task_type: "grounding",
      execution_trace: {
        router_decision: "grounding",
        models_called: ["router", "grounding"],
        processing_time_ms: 1340,
      },
      bounding_boxes: [
        { label: "building cluster", confidence: 0.88, x_min: 0.05, y_min: 0.08, x_max: 0.45, y_max: 0.4 },
        { label: "building cluster", confidence: 0.83, x_min: 0.55, y_min: 0.5, x_max: 0.95, y_max: 0.9 },
      ],
      metadata: { demo: true, source: "synthetic sample" },
    },
  },
  {
    id: "vegetation",
    label: "Vegetation",
    icon: "\u2698",
    query: "Describe this scene.",
    description: "Dense forest canopy with minimal human alteration.",
    image: scene({
      bg: "#16321f",
      blocks: [
        { x: 0, y: 0, w: 640, h: 480, fill: "#1e4429" },
        { x: 60, y: 60, w: 180, h: 140, fill: "#255530", opacity: 0.8 },
        { x: 320, y: 180, w: 220, h: 180, fill: "#255530", opacity: 0.8 },
        { x: 420, y: 40, w: 160, h: 120, fill: "#2c6338", opacity: 0.7 },
      ],
    }),
    mockResult: {
      query: "Describe this scene.",
      answer:
        "This is a densely vegetated area with continuous forest canopy and no visible built-up structures or bare soil. Canopy texture suggests mixed-density forest cover.",
      confidence: 0.86,
      task_type: "captioning",
      execution_trace: {
        router_decision: "captioning",
        models_called: ["geochat"],
        processing_time_ms: 870,
      },
      bounding_boxes: [],
      metadata: { demo: true, source: "synthetic sample" },
    },
  },
  {
    id: "change",
    label: "Change Analysis",
    icon: "\u21C4",
    query: "Are there signs of change?",
    description: "Scene with a partially cleared or altered region.",
    image: scene({
      bg: "#1e4429",
      blocks: [
        { x: 0, y: 0, w: 320, h: 480, fill: "#255530" },
        { x: 320, y: 0, w: 320, h: 480, fill: "#4a4331" },
      ],
    }),
    mockResult: {
      query: "Are there signs of change?",
      answer:
        "Yes. The eastern half of the scene shows cleared or bare ground contrasting sharply with intact vegetation to the west, consistent with recent land clearing.",
      confidence: 0.82,
      task_type: "change_detection",
      execution_trace: {
        router_decision: "change_detection",
        models_called: ["router", "grounding"],
        processing_time_ms: 1560,
      },
      bounding_boxes: [
        { label: "cleared area", confidence: 0.82, x_min: 0.5, y_min: 0.0, x_max: 1.0, y_max: 1.0 },
      ],
      metadata: { demo: true, source: "synthetic sample" },
    },
  },
];
