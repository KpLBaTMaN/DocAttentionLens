/**
 * A quiet "working" mark placed before status text while a model runs or attention is calculated.
 * Decorative only: the text beside it says what is happening, so screen readers skip it.
 */
export function Busy() {
  return <span className="docattentionlens-busy" aria-hidden="true" data-testid="busy"><span/><span/><span/></span>;
}

/** A thin line sweeping along the top edge of its positioned parent while a model run is in progress. */
export function BusySweep() {
  return <span className="docattentionlens-busy-sweep" aria-hidden="true" data-testid="busy-sweep"/>;
}
