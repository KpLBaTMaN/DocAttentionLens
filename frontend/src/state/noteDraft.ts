export type NoteDraft = {caseId:string;datasetVersion:string;baseRevision:number|null;name:string;notes:string};

export function loadNoteDraft(key:string,caseId:string,datasetVersion:string):NoteDraft|undefined {
  try {
    const value=JSON.parse(localStorage.getItem(key)??'null');
    if(value&&value.caseId===caseId&&value.datasetVersion===datasetVersion&&
      typeof value.name==='string'&&value.name.length<=200&&typeof value.notes==='string'&&value.notes.length<=20000&&
      (value.baseRevision===null||Number.isInteger(value.baseRevision))) return value;
  } catch {/* Saved server records remain usable when browser storage is unavailable. */}
  return undefined;
}
