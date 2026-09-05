// Scripted timeline for the frontend-only OASIS run simulation.
// Nothing here talks to a backend — it only drives the mock UI state.

export const TASKS = [
  {
    id: 'evidence',
    name: 'Pulling evidence',
    icon: '\u{1F6F0}\u{FE0F}',
    role: 'Grabs population, facility, and travel data for the area.',
  },
  {
    id: 'compile',
    name: 'Setting up the problem',
    icon: '\u{1F9E9}',
    role: 'Locks in a solvable version of the problem plus a starting plan.',
  },
  {
    id: 'search',
    name: 'Searching for fixes',
    icon: '\u{1F9ED}',
    role: 'Tries out changes to see what improves coverage.',
  },
  {
    id: 'validate',
    name: 'Double-checking',
    icon: '✅',
    role: "Rechecks every option before it's allowed to become the plan.",
  },
  {
    id: 'map',
    name: 'Drawing the map',
    icon: '\u{1F5FA}\u{FE0F}',
    role: 'Turns the verified plan into a map you can look at.',
  },
  {
    id: 'report',
    name: 'Writing the summary',
    icon: '\u{1F4CA}',
    role: 'Adds up the numbers — coverage, by group, by scenario.',
  },
];

export const STAGES = [
  {
    taskId: 'evidence',
    start: 0,
    end: 1800,
    toolCalls: 2,
    modelTokens: 180,
    generatedTokens: 0,
    blurbs: [
      'Finding the place and grabbing population numbers…',
      'Cleaning up units and facility candidates…',
      'Building the travel-time matrix…',
    ],
  },
  {
    taskId: 'compile',
    start: 1800,
    end: 3300,
    toolCalls: 1,
    modelTokens: 140,
    generatedTokens: 0,
    blurbs: [
      'Putting together a solvable version of this…',
      'Locking in a decent starting plan…',
    ],
  },
  {
    taskId: 'search',
    start: 3300,
    end: 8300,
    toolCalls: 3,
    modelTokens: 620,
    generatedTokens: 420,
    blurbs: [
      'Figuring out how to search given the budget…',
      'Trying: swap one site…',
      'Trying: swap a couple sites…',
      'Trying: reassign some areas…',
      'Trying: a scenario-aware tweak…',
    ],
  },
  {
    taskId: 'map',
    start: 8300,
    end: 9500,
    toolCalls: 1,
    modelTokens: 60,
    generatedTokens: 0,
    blurbs: ['Turning the verified plan into a map…'],
  },
  {
    taskId: 'report',
    start: 9500,
    end: 10300,
    toolCalls: 1,
    modelTokens: 40,
    generatedTokens: 0,
    blurbs: ['Adding up coverage by group and scenario…'],
  },
];

export const VALIDATE_BURSTS = [
  { start: 3800, end: 4300 },
  { start: 5100, end: 5600 },
  { start: 6400, end: 6900 },
  { start: 7700, end: 8200 },
];

export const TOTAL_MS = 10400;

export const DEFAULT_BUDGETS = {
  wallTimeS: 30,
  modelTokens: 4096,
  generatedTokens: 1024,
  toolCalls: 8,
};

export const TIMELINE = [
  { time: 0, kind: 'system', text: 'Got your prompt — starting the run.' },
  { time: 1800, kind: 'tool', text: "Evidence's in — population, facilities, and travel times are ready." },
  { time: 3300, kind: 'tool', text: 'Problem is locked in and ready to search.' },
  { time: 3320, kind: 'result', text: 'Starting plan: 0.612 coverage.' },
  { time: 4300, kind: 'tool', text: 'Trying a candidate: swap one site.' },
  { time: 4320, kind: 'result', text: "Checked out — coverage's up to 0.658." },
  { time: 5600, kind: 'tool', text: 'Trying a candidate: swap a couple sites.' },
  { time: 5620, kind: 'result', text: 'Checked out — up to 0.701.' },
  { time: 6900, kind: 'tool', text: 'Trying a candidate: reassign some areas.' },
  { time: 6920, kind: 'result', text: 'Checked out — up to 0.734.' },
  { time: 8200, kind: 'tool', text: 'Trying a candidate: scenario-aware tweak.' },
  { time: 8220, kind: 'result', text: 'Checked out — up to 0.758.' },
  { time: 8300, kind: 'system', text: "That's the search budget used up — no fallback needed, we're good." },
  { time: 9500, kind: 'tool', text: 'Map is ready.' },
  { time: 10300, kind: 'tool', text: 'Summary published — coverage by group and scenario.' },
  { time: 10400, kind: 'final', text: 'Done. Final plan: 0.758 coverage (up 23.9% from the starting plan).' },
];

export const EXAMPLES = [
  {
    id: 'cooling',
    label: 'Cooling-center coverage',
    prompt:
      'Figure out where to put cooling centers so they cover the most heat-vulnerable people, with a floor for the highest-risk group.',
  },
  {
    id: 'clinic',
    label: 'Clinic access',
    prompt:
      "Check how easy it is to reach a primary care clinic across the county, then improve it — and tell me how the worst-off group is doing.",
  },
  {
    id: 'mobile',
    label: 'Mobile vaccination routing',
    prompt:
      'Plan mobile vaccination routes that reach as many people as possible, within vehicle capacity and shift limits.',
  },
];
