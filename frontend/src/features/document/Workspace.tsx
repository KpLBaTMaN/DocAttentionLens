import { useCallback, useContext, useEffect, useRef, useState, type CSSProperties, type ReactNode } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { api, ApiError, errorMessage, unwrap, type Case, type Document, type Element, type Investigation, type DisplaySettings, type Method } from '../../api/client';
import { Message, ThemeContext } from '../../app/App';
import { HeaderSlot } from '../../app/HeaderSlot';
import { Busy, BusySweep } from '../../app/Busy';
import { useRemote } from '../../app/remote';
import { PageCanvas, type PageHit } from './PageCanvas';
import { RetrievalPanel, RetrievalDetails, SimilarityControls, SimilaritySource, type Capture } from './RetrievalPanel';
import { GenerationPanel, AnswerTokens, GenerationDetails, type Generation } from './GenerationPanel';
import { AttentionSource, GroupedComparison, MethodGrouping, MethodReadiness, MethodSettings } from './AttentionPanel';
import { CaptureMenu } from './CaptureMenu';
import { useRetrieverAttention, defaultSource, lastAnswerToken } from './useRetrieverAttention';
import { ContextAttention } from './ContextAttention';
import '../../styles/setup.css';
import { MethodResult, RetrieverResult, RetrieverRowDetails } from './MethodResult';
import { TokenStrip, decodePiece, displayPiece } from './TokenStrip';
import { SourceLine, SourceSegment, ImageChip, SourceChip } from './TokenExplorer';
import { PromptTokens, type PromptGhost, type PromptPart, type PromptToken } from './PromptTokens';
import { useAttentionColumns, columnValues } from './useAttentionColumns';
import type { Attention } from './GenerationPanel';
import { parametersFor, selectionKey, switchMethod, switchCapture, directionFor, openedView, RAW_ATTENTION, HEAD_MEAN_METHOD, allHeadsState, everyHead, singleHeadReturn } from './methodState';
import { ExperimentPanel } from './ExperimentPanel';
import { DocumentStudies } from './DocumentStudies';
import { AttentionInterventions } from './AttentionInterventions';
import { ElementInspector } from './ElementInspector';
import type { components } from '../../api/generated/schema';
import { loadNoteDraft, type NoteDraft } from '../../state/noteDraft';
import { useRetrievalAcquisition, useGenerationAcquisition, type UpdateView } from './useAcquisition';
import { DETAILS_WIDE_QUERY, Drawer, LEDGER_QUERY, readBrowserPreference, ResizeHandle, useBrowserChoice, useBrowserNumber, useBrowserPreference, useLeaveGuard, useMediaQuery, usePinnedHeight, useWideLayout } from './WorkspaceControls';
import { imageAttentionMaximum, attentionWeightLabel, attentionMode, rangeFloor, roundedWeightLabel, sharePercent, type AttentionMode } from './attentionDisplay';
import { scaleOptions } from './AttentionToolbar';
import { DisplaySettingsTab, isDisplayView, type DisplayView } from './DisplaySettingsPanel';
import { massBreakdown } from './massBreakdown';
import { PageKey, ScaleKey, rampStyle, scaleAgreement, strongestToken, type Ramp, type ScaleLine } from './ScaleKey';
import { IncomingWeights } from './IncomingWeights';
import { ResearchReceipt } from './ResearchReceipt';
import { RunProvenance } from './RunProvenance';
import { researchReceipt } from './receiptExport';
import { ElementSummary, type SourceReach } from './ElementSummary';
import { PageElements } from './PageElements';
import type { ElementRegions } from './PageCanvas';
import { contributions, elementMassMaximum, hasMappedElements, methodKey, NO_MAPPED_ELEMENTS, provenanceLabel, quantityLabel, remember, retrieverMappingGap } from './perspective';
import { analysisPerspective, attentionInputLabel, canGroupByElement, elementQuantityLabel, groupedDescriptor, selectedComparison } from './documentGrouping';
import { DocumentAttentionView } from './DocumentAttentionView';
import { useMethodSelection } from './useMethodSelection';
import { MethodPicker } from './MethodPicker';
import { MethodBar, type MethodStep } from './MethodBar';
import { LayerScrubber } from './LayerScrubber';
import { rolloutIntervalGap } from './rolloutInterval';
import { CanvasInstruments, CanvasLegend, ComparisonChoice, DetailsToggle, MethodSentence, ZoomPill } from './CanvasInstruments';
import { WorkspaceTopBar } from './WorkspaceTopBar';
import { TableDetails } from '../tables/TableDetails';
import { TableComparisons } from '../tables/TableComparisons';
import { FileText, Files, Maximize, Minimize, PanelRight, Sliders, X } from '../../app/icons';
import { familyLabel, filterChoices, groupByFamily, isElementProfile } from './methodChoices';
import { ElementProfile, ProfileHint, isProfileMetric, layerMeans, PROFILE_METRICS, type ProfileMetric, type ProfileStatus } from './ElementProfile';
import { perspectiveLabel } from './perspective';
import { DETAILS_TABS, DRAWER_TABS, DetailsPanel, isDetailsTab, type DetailsTab } from './DetailsPanel';
import { Filmstrip, type FilmstripPage } from './Filmstrip';
import { CommandPalette, ShortcutSheet, type Command } from './CommandPalette';
import { SHORTCUT_GROUPS, useWorkspaceShortcuts } from './useWorkspaceShortcuts';
import '../../styles/analytical.css';

type ViewState = components['schemas']['ViewState'];
type GenerationDetail='result'|'tokens'|'causal';
type RetrievalDetail='result'|'encoding'|'allocations'|'winners'|'contributions';
/** Details › Result view that shows the selected generator method's own output (element totals are in Details › Element). */
const detailFor=(methodId?:string|null):GenerationDetail=>!methodId||methodId==='attention.row'||methodId==='attention.element_allocation'?'causal':'result';
/** The same for a retriever encoding: element aggregation opens its linked element summary. */
const retrievalDetailFor=(methodId:string):RetrievalDetail=>methodId==='attention.row'?'encoding':methodId==='attention.element_allocation'?'allocations':'result';
const DEFAULT_VIEW: DisplaySettings = {schema_version:'0.2.0', zoom:1, center_x:null, center_y:null, show_boxes:true, show_support:false, show_reading_order:true, show_geometry_flags:true,input_view:'original',retrieval_scale_mode:'auto',retrieval_scale_max:1,attention_scale_mode:'auto',attention_scale_max:1,attention_opacity:.65,attention_display_mode:'overlay',attention_show_patches:false,attention_smooth:false,attention_colormap:'inferno',attention_range_floor:'zero',attention_grouping:'positions',retrieval_grouping:'positions',attention_element_view:'elements',attention_grouped_comparison_index:0,comparison_columns:2,comparison_zoom:1,comparison_center_x:null,comparison_center_y:null,similarity_opacity:.65,rail_width:264,inspector_width:360,dock_height:180,rail_collapsed:false,inspector_collapsed:false,rail_tab:'pages',evidence_dock_open:false,canvas_fit:'manual'};
/** The element ledger's width (the saved `rail_width`), kept readable beside the page. */
const LEDGER_MIN=240, LEDGER_MAX=440;

export function Workspace({record, document, saved, initialCapture, initialGeneration, initialLens='generation', initialFocus='both', initialRole='generation'}: {initialRole?:'generation'|'retrieval';initialLens?:'generation'|'context';initialFocus?:'generation'|'context'|'both';record:Case; document:Document; saved?:Investigation; initialCapture?:Capture; initialGeneration?:Generation}) {
  const navigate = useNavigate();
  const [investigation, setInvestigation] = useState(saved);
  const [state, setState] = useState<ViewState>(()=>openedView({similarity_direction:'row',retrieval_attention:true,retrieval_encoding_id:initialCapture?.attention_encodings?.find(e=>e.pass_kind!=='query')?.encoding_id??'document',retrieval_source_index:defaultSource(initialCapture?.attention_encodings?.find(e=>e.pass_kind!=='query')),retrieval_attention_method:'attention.row',retrieval_comparison_index:0,...saved?.view_state,investigation_model_role:(!saved?.view_state?.generation_capture_id&&saved?.view_state?.retrieval_capture_id&&saved?.view_state?.evidence_signal==='retrieval')?'retrieval':saved?.view_state?.investigation_model_role??initialRole,attention_lens:saved?.view_state?.attention_lens??initialLens,investigation_focus:saved?.view_state?.investigation_focus??initialFocus,attention_method_id:saved?.view_state?.attention_method_id??'attention.row',attention_variant_id:saved?.view_state?.attention_variant_id??'captured_weights_v1',page_id:saved?.view_state?.page_id??initialGeneration?.context.selected_page_ids[0]??initialCapture?.attention_encodings?.find(e=>e.pass_kind==='document')?.page_id??initialCapture?.pages[0].page_id??record.page_ids[0],query_position:saved?.view_state?.query_position??0,similarity_overlay:saved?.view_state?.similarity_overlay??true,retrieval_capture_id:saved?.view_state?.retrieval_capture_id??initialGeneration?.retrieval_capture_id??initialCapture?.capture_id??null,generation_capture_id:saved?.view_state?.generation_capture_id??initialGeneration?.capture_id??null,evidence_signal:saved?.view_state?.evidence_signal??(initialGeneration?'attention':initialCapture?'retrieval':'document'),attention_step:saved?.view_state?.attention_step??0,attention_layer:saved?.view_state?.attention_layer??0,attention_head:saved?.view_state?.attention_head??0,attention_overlay:saved?.view_state?.attention_overlay??true,attention_direction:directionFor(saved?.view_state?.investigation_model_role==='retrieval'?saved.view_state.retrieval_attention_method:saved?.view_state?.attention_method_id,saved?.view_state?.attention_direction??'row')}));
  // A new investigation opens with Details closed (or as last left on a wide screen). Saved investigations
  // reopen as saved: an open inspector reopens Details, and an open evidence dock reopens it on Result.
  const [settings, setSettings] = useState<DisplaySettings>(()=>{
    const opened={...DEFAULT_VIEW,...(!saved?{inspector_collapsed:!(window.matchMedia(DETAILS_WIDE_QUERY).matches&&readBrowserPreference('docattentionlens-details-open',false))}:{}),...saved?.display_settings};
    return saved?.display_settings?.evidence_dock_open?{...opened,inspector_collapsed:false}:opened;
  });
  const [drawer,setDrawer]=useState<'acquire'|'experiment'|'research'|'interventions'|null>(null);
  const [acquisitionSignal,setAcquisitionSignal]=useState<'attention'|'retrieval'>('retrieval');
  const [imageFocus,setImageFocus]=useState(false);
  const [paletteOpen,setPaletteOpen]=useState(false);
  const [shortcutsOpen,setShortcutsOpen]=useState(false);
  const detailsWide=useMediaQuery(DETAILS_WIDE_QUERY);
  const [,setDetailsPreference]=useBrowserPreference('docattentionlens-details-open',false);
  // The Details tab is a per-browser preference; a saved open dock or element rail opens on Result or Element.
  const [preferredTab,setPreferredTab]=useBrowserChoice<DetailsTab>('docattentionlens-details-tab','result',isDetailsTab);
  const [detailsTab,setDetailsTabState]=useState<DetailsTab>(()=>saved?.display_settings?.evidence_dock_open?'result':saved?.display_settings?.rail_tab==='elements'&&saved.view_state?.selected_element_id?'element':preferredTab);
  const [wideTable,setWideTable]=useState(false);
  // The method panel under the method bar (three-column layout): what the method reads, its settings and the list.
  const [methodPanelOpen,setMethodPanelOpen]=useState(false);
  const {theme,setTheme}=useContext(ThemeContext);
  const [canvasAction,setCanvasAction]=useState<{id:number;box:Element['display_bbox']}|undefined>();
  const [retrievalDetail,setRetrievalDetail]=useState<RetrievalDetail>(saved?.view_state?.investigation_model_role==='retrieval'&&(saved.view_state.retrieval_attention_method??'attention.row')!=='attention.row'?retrievalDetailFor(saved.view_state.retrieval_attention_method!):'winners');
  const [generationDetail,setGenerationDetail]=useState<GenerationDetail>(detailFor(saved?.view_state?.attention_method_id));
  // Phones show Details and the pages as bottom sheets over the bottom bar.
  const [sheet,setSheet]=useState<'details'|'pages'|null>(null);
  const wide=useWideLayout();
  const ledgerWide=useMediaQuery(LEDGER_QUERY);
  // The source line shows question and answer tokens; the full prompt (template tokens included) is its other view.
  // The full prompt can be hidden to its header and its rows resized; both are per-browser conveniences.
  const [fullPrompt,setFullPrompt]=useBrowserPreference('docattentionlens-full-prompt',false);
  const [promptHidden,setPromptHidden]=useBrowserPreference('docattentionlens-prompt-hidden',false);
  const [promptHeight,setPromptHeight]=useBrowserNumber('docattentionlens-prompt-height');
  // Details › Display rounds values for a quick read; Exact restores the calculated values.
  const [exactScale,setExactScale]=useBrowserPreference('docattentionlens-scale-exact',false);
  // Details › Display shows its settings or the colour scale readout; the viewer's last choice is kept.
  const [displayView,setDisplayView]=useBrowserChoice<DisplayView>('docattentionlens-display-view','settings',isDisplayView);
  const scaleWeight=exactScale?attentionWeightLabel:roundedWeightLabel;
  const [pageHit,setPageHit]=useState<PageHit>();
  // Pinned bands publish their heights, so sticky offsets and the fitted page height stay exact when bands wrap.
  const topBar=useRef<HTMLElement|null>(null), sourceLine=useRef<HTMLDivElement>(null), scrubberBand=useRef<HTMLDivElement>(null), messages=useRef<HTMLDivElement>(null), bottomBar=useRef<HTMLElement>(null);
  if(!topBar.current&&typeof window!=='undefined')topBar.current=window.document.querySelector<HTMLElement>('.docattentionlens-header');
  usePinnedHeight(topBar,'--docattentionlens-top-h');
  usePinnedHeight(sourceLine,'--docattentionlens-source-h');
  usePinnedHeight(scrubberBand,'--docattentionlens-scrubber-h');
  usePinnedHeight(messages,'--docattentionlens-messages-h');
  usePinnedHeight(bottomBar,'--docattentionlens-nav-h');
  const imageViews=useRef<HTMLDivElement>(null), inspectorPanel=useRef<HTMLElement>(null), elementPanel=useRef<HTMLElement>(null), elementSummarySection=useRef<HTMLElement>(null);
  /** Brings the page back under the pinned bands when it has scrolled out of view. */
  const revealPage=()=>{const top=imageViews.current?.getBoundingClientRect().top;if(top!==undefined&&top<(topBar.current?.offsetHeight??0)+(sourceLine.current?.offsetHeight??0))imageViews.current?.scrollIntoView({block:'start'});};
  // Clicking a generator patch shows its incoming raw column; the next token click restores the method it replaced.
  const [rowMethod,setRowMethod]=useState<Pick<Method,'method_id'|'variant_id'>>();
  // The scrubber's All cell shows the head mean; leaving it returns to the single-head method it was chosen from.
  const [headMethod,setHeadMethod]=useState<string>();
  const [name, setName] = useState(saved?.name ?? document.filename.replace(/\.pdf$/i,''));
  const [notes, setNotes] = useState(saved?.notes ?? '');
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>();
  const [conflict, setConflict] = useState(false);
  const draftKey=`docattentionlens-note-draft:${saved?.investigation_id??record.case_id}`;
  const [draft,setDraft]=useState<NoteDraft|undefined>(()=>loadNoteDraft(draftKey,record.case_id,record.dataset_version));
  const pageId = state.page_id ?? record.page_ids[0];
  const previousPage=useRef(pageId);
  useEffect(()=>{if(previousPage.current!==pageId){setSettings(v=>({...v,canvas_fit:'page',zoom:1,center_x:null,center_y:null}));previousPage.current=pageId;}},[pageId]);
  const capture = useRemote(useCallback(async()=>state.retrieval_capture_id?unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}',{params:{path:{capture_id:state.retrieval_capture_id}}})):undefined,[state.retrieval_capture_id]));
  // MaxSim opens on the first question word: position 0 is <bos>, which always matches the page's <bos>.
  // Saved views keep their recorded position.
  const openedCapture=useRef(saved?.view_state?.retrieval_capture_id??null);
  useEffect(()=>{
    const data=capture.data;
    if(!data||openedCapture.current===data.capture_id)return;
    openedCapture.current=data.capture_id;
    const word=data.query_positions.find(p=>p.eligible&&p.role==='query_text');
    if(word)setState(s=>s.retrieval_capture_id===data.capture_id&&(s.query_position??0)===0?{...s,query_position:word.index}:s);
  },[capture.data]);
  const generation=useRemote(useCallback(async()=>state.generation_capture_id?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}',{params:{path:{capture_id:state.generation_capture_id}}})):undefined,[state.generation_capture_id]));
  // Table questions record accepted answers; the server applies the exact-match rubric (evaluation only).
  const answerCheck=useRemote(useCallback(async()=>record.table_grounding&&state.generation_capture_id?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}/answer-check',{params:{path:{capture_id:state.generation_capture_id}}})):undefined,[record.table_grounding,state.generation_capture_id]));
  const activeMethod=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/methods/{method_id}',{params:{path:{method_id:state.attention_method_id??'attention.row'},query:{variant_id:state.attention_variant_id??'captured_weights_v1'}}})),[state.attention_method_id,state.attention_variant_id]));
  const legacyRow=state.attention_method_id==='attention.row'||state.attention_method_id==='attention.element_allocation';
  const legacyAttention=useRemote(useCallback(async()=>state.generation_capture_id&&legacyRow?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}/rows',{params:{path:{capture_id:state.generation_capture_id},query:{step:state.attention_step??0,layer:state.attention_layer??0,head:state.attention_head??0,...(state.attention_source_index!=null?{source_index:state.attention_source_index}:{})}}})):undefined,[state.generation_capture_id,state.attention_step,state.attention_layer,state.attention_head,state.attention_source_index,legacyRow]));
  const kSupport=generation.data?.source_rows?.find(r=>r.source_index===state.attention_source_index)?.target_count??generation.data?.steps.find(s=>s.step===(state.attention_step??0))?.target_count;
  const kRequested=state.attention_parameters?.k;
  const kExceeds=!!(activeMethod.data?.parameters_schema.properties as Record<string,unknown>|undefined)?.k&&typeof kRequested==='number'&&kSupport!==undefined&&kRequested>kSupport;
  // An invalid saved rollout interval waits for repair in the layer scrubber; it is never requested.
  const intervalGap=generation.data?rolloutIntervalGap(state.attention_method_id??undefined,generation.data.layer_count,state.attention_layer??0,state.attention_parameters?.start_layer??0):undefined;
  // Table-aware comparisons read other saved captures of questions on this table: the question-swap
  // baseline (Lookup-path allocation) and the matched question (Matched-question contrast).
  const tableRelated=useRemote(useCallback(async()=>record.table_grounding&&state.generation_capture_id?unwrap(await api.GET('/api/v1/generation/captures/{capture_id}/table-related',{params:{path:{capture_id:state.generation_capture_id}}})):undefined,[record.table_grounding,state.generation_capture_id]));
  const [tableBaseline,setTableBaseline]=useState(true);
  const [tablePartner,setTablePartner]=useState<string>();
  const [tableCell,setTableCell]=useState<string>();
  const partnerId=tableRelated.data?.matched_capture_ids.includes(tablePartner??'')?tablePartner:tableRelated.data?.matched_capture_ids[0];
  const tableExtras=state.attention_source_index!=null?{}
    :activeMethod.data?.method_id==='table.lookup_path'&&tableBaseline&&tableRelated.data?.baseline_capture_ids.length?{sample_capture_ids:[state.generation_capture_id!,...tableRelated.data.baseline_capture_ids]}
    :activeMethod.data?.method_id==='table.matched_contrast'&&partnerId?{comparison_capture_id:partnerId}:{};
  const waitingForTable=(activeMethod.data?.method_id==='table.lookup_path'||activeMethod.data?.method_id==='table.matched_contrast')&&!tableRelated.data&&!tableRelated.error;
  const analysisBody=!kExceeds&&!intervalGap&&!waitingForTable&&state.generation_capture_id&&activeMethod.data&&activeMethod.data.method_id===state.attention_method_id&&activeMethod.data.variant_id===state.attention_variant_id?{generation_capture_id:state.generation_capture_id,method_id:activeMethod.data.method_id,variant_id:activeMethod.data.variant_id,parameters:parametersFor(activeMethod.data,state),...(state.attention_source_index!=null?{source_index:state.attention_source_index}:{}),...tableExtras}:undefined;
  const requestKey=JSON.stringify(analysisBody);
  const latestAnalysis=useRef(requestKey);
  latestAnalysis.current=requestKey;
  const methodAnalysis=useRemote(useCallback(async()=>{
    if(!analysisBody)return undefined;
    await new Promise(resolve=>setTimeout(resolve,150));
    // A selection superseded during the pause (e.g. several head boxes unticked) sends nothing.
    if(latestAnalysis.current!==requestKey)return undefined;
    return {key:requestKey,result:unwrap(await api.POST('/api/v1/methods/analyses',{body:analysisBody}))};
  // The serialized request is the complete scientific selection, including parameter drafts.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[requestKey]));
  const analysis=methodAnalysis.data&&methodAnalysis.data.key===requestKey?methodAnalysis.data.result:undefined;
  const selectedSource=state.attention_source_index??generation.data?.steps.find(p=>p.step===(state.attention_step??0))?.source_index;
  const legacyCurrent=legacyAttention.data?.capture_id===state.generation_capture_id&&legacyAttention.data?.source_index===selectedSource&&(state.attention_source_index!=null||legacyAttention.data?.step===state.attention_step)&&legacyAttention.data?.layer===state.attention_layer&&legacyAttention.data?.head===state.attention_head;
  const groupedComparison=canGroupByElement(activeMethod.data)&&analysisPerspective(activeMethod.data,settings,'generation')==='document_aware'
    ? selectedComparison(analysis?.comparisons,settings.attention_grouped_comparison_index) : undefined;
  // A method that returns one map per layer, head or condition (Attention Across Layers, Head-by-Head) has no single
  // row: the page draws the chosen map, on the scale shared by every map, and Details › Result shows them all.
  const shownComparison=groupedComparison??(analysis&&!analysis.row?selectedComparison(analysis.comparisons,settings.attention_grouped_comparison_index):undefined);
  const attention={data:legacyRow?(legacyCurrent?legacyAttention.data:undefined):analysis?.row??shownComparison?.row??undefined,error:legacyRow?legacyAttention.error:methodAnalysis.error};
  useEffect(()=>{const key=selectionKey(state);if(analysis)setState(s=>selectionKey(s)!==key||s.attention_analysis_id===analysis.analysis_id?s:{...s,attention_analysis_id:analysis.analysis_id});},[analysis]);
  const capturedPage = capture.data?.pages.find(p=>p.page_id===pageId);
  const row = useRemote(useCallback(async()=>state.retrieval_capture_id&&capturedPage?unwrap(await api.GET('/api/v1/retrieval/captures/{capture_id}/rows',{params:{path:{capture_id:state.retrieval_capture_id},query:{page_id:pageId,query_index:state.query_position??0}}})):undefined,[state.retrieval_capture_id,state.query_position,pageId,capturedPage]));
  const changeEvidence:UpdateView = useCallback(next=>{setState(previous=>{const proposed=typeof next==='function'?next(previous):next;const resolved={...switchCapture(previous,proposed,activeMethod.data)};if(resolved.investigation_model_role==='retrieval'&&resolved.evidence_signal==='attention'){if(resolved.generation_capture_id)resolved.investigation_model_role='generation';else resolved.evidence_signal='retrieval';}return selectionKey(previous)!==selectionKey(resolved)?{...resolved,attention_analysis_id:null}:resolved;});setDirty(true);},[activeMethod.data]);
  const captureModel=capture.data?.provenance?.model_registration_id;
  const retrievalSession=useRetrievalAcquisition(record,changeEvidence,state.investigation_model_role==='retrieval',typeof captureModel==='string'?captureModel:'builtin-retrieval');
  const generationSession=useGenerationAcquisition(record,capture.data,changeEvidence);
  const isRetrievalInvestigation=state.investigation_model_role==='retrieval';
  const isGeneration=!isRetrievalInvestigation&&state.evidence_signal!=='retrieval';
  const consumed=generation.data?.context.pages.find(p=>p.page_id===pageId);
  const prediction=generation.data?.steps.find(p=>p.step===(state.attention_step??0));
  const retrieverAttention=useRetrieverAttention(capture.data,state);
  // Saved-capture availability plus the case's verified document structure and its provenance.
  const prepared=useRemote(useCallback(async()=>unwrap(await api.GET('/api/v1/methods/availability',{params:{query:{case_id:record.case_id,
    ...(isRetrievalInvestigation?{}:{generation_capture_id:state.generation_capture_id??undefined,retrieval_capture_id:state.retrieval_capture_id??undefined})}}})),
    [record.case_id,isRetrievalInvestigation,state.generation_capture_id,state.retrieval_capture_id]));
  const structure=prepared.data?.structure??undefined;
  // Grouping is independent of the canonical calculation descriptor and request.
  const activePrepared=prepared.data?.items.find(p=>p.method.method_id===state.attention_method_id&&p.method.variant_id===state.attention_variant_id);
  // The availability list is already loaded, so a method switch never shows the other perspective while its descriptor loads.
  const activeDescriptor=isRetrievalInvestigation?retrieverAttention.method:activePrepared?.method??(activeMethod.data?.method_id===state.attention_method_id?activeMethod.data:undefined);
  const activePerspective=analysisPerspective(activeDescriptor,settings,isRetrievalInvestigation?'retrieval':'generation');
  const documentGap=activePerspective!=='document_aware'?undefined:isRetrievalInvestigation?retrieverAttention.unavailable
    ??retrieverMappingGap(activeDescriptor&&groupedDescriptor(activeDescriptor),retrieverAttention.encoding,retrieverAttention.page?.ownership,state.retrieval_source_index??0)
    :activePrepared&&!activePrepared.available?activePrepared.reasons.join(' ')
    :generation.data&&!hasMappedElements(generation.data.positions.slice(0,attention.data?.target_count).map(p=>p.ownership))?NO_MAPPED_ELEMENTS:undefined;
  const retrieverMethod=state.retrieval_attention_method??'attention.row';
  const previousRetrieverMethod=useRef(retrieverMethod);
  useEffect(()=>{if(previousRetrieverMethod.current===retrieverMethod)return;previousRetrieverMethod.current=retrieverMethod;
    setRetrievalDetail(retrievalDetailFor(retrieverMethod));
    // An open Details panel turns to the method's own output; a closed one stays closed (the canvas notice links to it).
    const groupedInput=activePerspective==='document_aware'&&canGroupByElement(activeDescriptor);
    if(retrieverMethod!=='attention.row'&&retrieverMethod!=='attention.element_allocation'&&!groupedInput)showResultIfOpen();if(retrieverMethod!=='attention.row')setDirty(true);
  // Only a method change turns Details to Result; layout and catalogue loading do not.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[retrieverMethod]);
  const hasAttention=isGeneration||retrieverAttention.enabled;
  // Switching to Document-aware opens Details on Element, unless the switch was made in Details › Method.
  const previousPerspective=useRef<string|undefined>(undefined);
  useEffect(()=>{if(!activeDescriptor)return;const previous=previousPerspective.current;previousPerspective.current=activePerspective;
    if(previous!==undefined&&previous!==activePerspective&&activePerspective==='document_aware'&&hasAttention&&!methodsOpen&&!ledgerLayout)openDetails('element',{sheet:false});
  // Only the perspective change itself opens Details.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  },[activePerspective,hasAttention,!!activeDescriptor]);
  const columns=useAttentionColumns({state,generation:generation.data,capture:capture.data,pageId,generatorActive:isGeneration,similarityActive:!hasAttention});
  // Only a generator patch shows what attends to it: it has no captured row of its own.
  const attentionColumnMode=isGeneration&&state.attention_direction==='column';
  const similarityColumnMode=!hasAttention&&state.similarity_direction==='column';
  const columnMode=attentionColumnMode||similarityColumnMode;
  const generatorColumn=columnValues(columns.generator.data);
  const retrieverPositions=(retrieverAttention.page?.positions??[]).map((p,i)=>({...p,role:(p.bbox?'image':'special') as 'image'|'special',page_id:retrieverAttention.page!.page_id,ownership:retrieverAttention.page!.ownership[i]}));
  const retrieverDisplay=retrieverAttention.row?.values;
  // A patch attends strongly to itself: the source patch would set the whole scale.
  const retrieverExcluded=retrieverAttention.row?.source_index;
  const comparisonRows=retrieverAttention.result?.comparisons;
  const retrieverImageMax=Math.max(...(comparisonRows?.length?comparisonRows.flatMap(c=>retrieverPositions.filter(p=>p.bbox&&p.index!==retrieverExcluded&&p.index<c.row.target_count).map(p=>c.row.values[p.index]??0)):retrieverPositions.filter(p=>p.bbox&&p.index!==retrieverExcluded).map(p=>retrieverDisplay?.[p.index]??0)),1e-8);
  const sourcePatchExcluded=!!retrieverPositions.find(p=>p.index===retrieverExcluded)?.bbox;
  const scaleMode=hasAttention?settings.attention_scale_mode??'auto':settings.retrieval_scale_mode??'auto';
  const imageMax=analysis?.comparisons?.length?Math.max(...analysis.comparisons.flatMap(c=>(generation.data?.positions??[]).filter(p=>p.role==='image'&&p.index<c.row.target_count).map(p=>c.row.values[p.index])),1e-8):imageAttentionMaximum(generation.data,attention.data);
  const autoMax=retrieverAttention.enabled?(scaleMode==='causal'||retrieverAttention.encoding?.pass_kind==='query'?(comparisonRows?.length?Math.max(...comparisonRows.map(c=>c.row.display_max)):retrieverAttention.row?.display_max??1):retrieverImageMax):isGeneration?(scaleMode==='causal'?(analysis?.comparisons?.length?Math.max(...analysis.comparisons.map(c=>c.row.display_max)):attention.data?.display_max??1):imageMax):Math.max(...(row.data?.similarities??[]).map(v=>Math.abs(v??0)),1e-8);
  const scaleMax=scaleMode==='fixed'?(hasAttention?settings.attention_scale_max??1:settings.retrieval_scale_max??1):autoMax;
  const page = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/pages/{page_id}', {params:{path:{page_id:pageId}}})), [pageId]));
  const elements = useRemote(useCallback(async () => unwrap(await api.GET('/api/v1/pages/{page_id}/elements', {params:{path:{page_id:pageId}}})), [pageId]));
  // Table grounding cases annotate a table on the page (cells, rows, columns), not layout elements.
  const tableCase = !!record.table_grounding;
  // Wide windows use the three-column layout; table grounding cases and narrower windows keep tabbed Details.
  const ledgerLayout=wide&&ledgerWide&&!tableCase;
  useEffect(()=>{if(!ledgerLayout)setMethodPanelOpen(false);},[ledgerLayout]);
  const tables = useRemote(useCallback(async () => tableCase ? unwrap(await api.GET('/api/v1/pages/{page_id}/tables', {params:{path:{page_id:pageId}}})) : undefined, [pageId, tableCase]));
  const selected = elements.data?.items.find(e=>e.element_id===state.selected_element_id);
  const source = useRemote(useCallback(async () => state.source_element_id ? unwrap(await api.GET('/api/v1/elements/{element_id}', {params:{path:{element_id:state.source_element_id}}})) : undefined, [state.source_element_id]));
  const target = useRemote(useCallback(async () => state.target_element_id ? unwrap(await api.GET('/api/v1/elements/{element_id}', {params:{path:{element_id:state.target_element_id}}})) : undefined, [state.target_element_id]));
  const support = new Set(record.support_element_ids);
  const ordered = [...(elements.data?.items ?? [])].sort((a,b)=>(a.reading_order ?? Number.MAX_SAFE_INTEGER)-(b.reading_order ?? Number.MAX_SAFE_INTEGER));
  /** Result tables can link elements on other pages; those are fetched, then selected (which also turns to their page). */
  async function chooseById(id:string) {
    const local=elements.data?.items.find(e=>e.element_id===id);
    if(local)return choose(local);
    try {choose(unwrap(await api.GET('/api/v1/elements/{element_id}',{params:{path:{element_id:id}}})));} catch(reason) {setError(errorMessage(reason));}
  }
  /**
   * Choosing an element (on the page, in a list or a result table) opens Details on Element; on phones, its sheet.
   * The three-column layout shows it in the element inspector, and an open Details drawer stays open.
   */
  function choose(element:Element) {
    setState(previous=>({...previous,page_id:element.page_id,selected_element_id:element.element_id}));setDirty(true);
    if(!ledgerLayout)openDetails('element',{focus:false});
    // The selected element's values lead the Element tab; bring them into view in the panel.
    requestAnimationFrame(()=>requestAnimationFrame(()=>(ledgerLayout?elementPanel:inspectorPanel).current?.querySelector('.docattentionlens-element-inspection')?.scrollIntoView({block:'nearest'})));
  }
  /** MaxSim is not attention: its controls and ledgers are in Details › Result. */
  function showSimilarity() {changeEvidence(s=>({...s,evidence_signal:'retrieval',...(s.investigation_model_role==='retrieval'?{retrieval_attention:false}:{})}));setRetrievalDetail('winners');openDetails('result',{focus:false});}
  function changePage(id:string) {setState(previous=>({...previous,page_id:id,selected_element_id:null}));setSettings(previous=>({...previous,canvas_fit:'page',zoom:1,center_x:null,center_y:null}));setDirty(true);requestAnimationFrame(revealPage);}
  function changeView(next:DisplaySettings) {setSettings(next);setDirty(true);}
  /** On a full-attention capture, full-matrix methods may read an image patch as their source. */
  const fullAttention=generation.data?.full_attention??undefined;
  // The same run (pages, model, prompt and page source) captured with full attention, offered where methods need it.
  const shown=generation.data;
  const fullAttentionTwin=shown&&!fullAttention?generationSession.records.data?.items.find(c=>!!c.full_attention&&c.capture_id!==shown.capture_id
    &&c.retrieval_capture_id===shown.retrieval_capture_id&&c.context.system_prompt===shown.context.system_prompt
    &&c.provenance.model_registration_id===shown.provenance.model_registration_id
    &&c.context.selected_page_ids.join('|')===shown.context.selected_page_ids.join('|')):undefined;
  const fullMatrixMethod=(m?:Pick<Method,'requires'>|null)=>!!fullAttention&&!!m?.requires.some(r=>r==='full_square_attention_matrices'||r==='captured_image_sources');
  const imageSource=(index?:number|null)=>index!=null&&generation.data?.positions[index]?.role==='image';
  function sourceFor(s:ViewState,method:Method) {
    if(!fullMatrixMethod(method))return imageSource(s.attention_source_index)?null:s.attention_source_index;
    if(!method.requires.includes('captured_image_sources')||imageSource(s.attention_source_index))return s.attention_source_index;
    // A patch reaches only earlier positions, so the page's last patch is the first source that sees every patch on it.
    return generation.data?.positions.filter(p=>p.role==='image'&&p.page_id===s.page_id&&p.index<fullAttention!.square_limit).at(-1)?.index??s.attention_source_index;
  }
  /** `preserveAxes` keeps the prediction/source, layer and head, as a perspective switch requires. */
  function selectMethod(method:Method,{preserveAxes=false,grouped=false}:{preserveAxes?:boolean;grouped?:boolean}={}) {
    // Each perspective remembers its last method, including the one being left.
    const departing=activeMethod.data&&methodKey(activeMethod.data)===`${state.attention_method_id}/${state.attention_variant_id}`?activeMethod.data:undefined;
    setRowMethod(undefined);
    changeEvidence(s=>({...switchMethod(s,method,activeMethod.data,{preserveAxes}),attention_source_index:sourceFor(s,method),evidence_signal:'attention',attention_direction:directionFor(method.method_id,s.attention_direction),
      perspective_methods:remember(remember(s.perspective_methods,'generation',departing),'generation',method)}));
    setGenerationDetail(detailFor(method.method_id));
    // A grouped input's output is the element summary (Details › Element), so the tab stays.
    if(!grouped&&method.method_id!=='attention.row'&&method.method_id!=='attention.element_allocation')showResultIfOpen();
  }
  function openDrawer(next:typeof drawer,signal:'attention'|'retrieval'=isGeneration?'attention':'retrieval') {if(next==='acquire')setAcquisitionSignal(signal);setDrawer(next);}
  /** Page + question with the reference answer (opt-in, not in this capture): retrieval acquisition for the page on screen with that pass chosen. */
  function acquireAnswerReading() {
    retrievalSession.setCaptureJointAnswer(true);
    retrievalSession.setAttentionPage(state.page_id??record.page_ids[0]);
    if(state.page_id)retrievalSession.setCandidates(pages=>pages.includes(state.page_id!)?pages:[state.page_id!]);
    openDrawer('acquire','retrieval');
  }
  /** Image focus hides the filmstrip, the element columns, Details and the source line; the page keeps its instruments and the scrubber. */
  function toggleImageFocus() {setImageFocus(v=>!v);setSheet(null);setWideTable(false);setMethodPanelOpen(false);}
  /** The method's own output: Details › Result. */
  function showEvidence() {openDetails('result');}
  useLeaveGuard(dirty);
  useEffect(()=>{
    if(!dirty) return;
    const baselineName=investigation?.name??document.filename.replace(/\.pdf$/i,'');
    if(name===baselineName&&notes===(investigation?.notes??'')) return;
    try {localStorage.setItem(draftKey,JSON.stringify({caseId:record.case_id,datasetVersion:record.dataset_version,baseRevision:investigation?.revision??null,name,notes} satisfies NoteDraft));} catch {/* Leave warning still protects in-memory edits. */}
  },[dirty,draftKey,name,notes,investigation?.revision,investigation?.name,investigation?.notes,record.case_id,record.dataset_version,document.filename]);
  async function save() {
    setSaving(true);setError(undefined);setConflict(false);
    try {
      const body={name,notes,case_id:record.case_id,view_state:!state.generation_capture_id&&state.evidence_signal==='attention'?{...state,evidence_signal:'document' as const}:state,display_settings:settings};
      const next=investigation ? unwrap(await api.PATCH('/api/v1/investigations/{investigation_id}', {params:{path:{investigation_id:investigation.investigation_id}},body:{...body,expected_revision:investigation.revision}})) : unwrap(await api.POST('/api/v1/investigations',{body}));
      setInvestigation(next);setDirty(false);
      setDraft(undefined);
      try {localStorage.removeItem(draftKey);} catch {/* Server save already succeeded. */}
      if(!investigation) navigate(`/attention-tool/${next.investigation_id}`,{replace:true});
    } catch(reason) {setError(errorMessage(reason));setConflict(reason instanceof ApiError&&reason.code==='revision_conflict');} finally {setSaving(false);}
  }
  function canvas(input = false) {
    if (!page.data || !elements.data) return <Message>{page.error ?? elements.error ?? 'Loading page and annotations…'}</Message>;
    return <PageCanvas key={`${pageId}/${input?'consumed':'original'}`} page={page.data} elements={elements.data.items} settings={settings} selected={state.selected_element_id??null} source={state.source_element_id??null} target={state.target_element_id??null} support={support} onView={changeView} scaleMax={scaleMax} focusAction={canvasAction}
      imageUrl={input?consumed?.preview_url:undefined} inputLabel={input?'Consumed input mapped to source page':'Original page'}
      evidence={!hasAttention&&!similarityColumnMode&&state.similarity_overlay!==false&&capturedPage&&row.data?{positions:capturedPage.positions,row:row.data}:undefined}
      pick={pickable} onElement={pickable?undefined:choose}
      attention={pageAttention} attentionMode={hasAttention?displayMode:'overlay'} regions={regions} onHover={setPageHit}
      tables={tables.data?{items:tables.data.items,grounding:record.table_grounding,highlight:tableCell}:undefined}/>;
  }
  const draftNotice=draft&&<Message>A compatible unsaved note draft is available{draft.baseRevision!==null&&draft.baseRevision!==investigation?.revision?' from an earlier revision':''}. <button className="docattentionlens-button" onClick={()=>{setName(draft.name);setNotes(draft.notes);setDraft(undefined);setDirty(true);}}>Restore note draft</button><button className="docattentionlens-button" onClick={()=>{setDraft(undefined);try{localStorage.removeItem(draftKey);}catch{/* Optional draft. */}}}>Discard note draft</button></Message>;
  const inputView=isGeneration&&consumed&&generation.data?.context.representation!=='ocr'&&generation.data?.context.representation!=='structured'?(settings.input_view??'original'):'original';
  // Details is the right panel: closed by default, a bottom sheet on phones, hidden in image focus.
  // `rail_collapsed` and `dock_height` stay in saved settings because the schema requires them; the UI never reads them.
  const detailsOpen=imageFocus?false:wide?!(settings.inspector_collapsed??false):sheet==='details';
  const fitMode=settings.canvas_fit==='width'?'width':(settings.zoom??1)===1&&settings.center_x===null&&settings.center_y===null?'page':'manual';
  const generationView:Exclude<GenerationDetail,'result'>|'result'=generationDetail==='result'&&legacyRow?'causal':generationDetail;
  const retrievalView:RetrievalDetail=retrieverAttention.enabled?retrievalDetail:retrievalDetail==='winners'||retrievalDetail==='contributions'?retrievalDetail:'winners';
  const retrieverOnPage=retrieverAttention.page?.page_id===pageId;
  // One line under the page names the row being shown; numbers and results stay in the panel and dock.
  // A retained question row, or on a full-attention capture any prompt token's row (read from the saved triangle).
  const promptSource=(index?:number|null)=>index!=null&&!!fullAttention&&index<fullAttention.position_count&&!imageSource(index)?generation.data?.positions[index]:undefined;
  const retainedRow=generation.data?.source_rows?.find(r=>r.source_index===state.attention_source_index);
  const promptToken=retainedRow?undefined:promptSource(state.attention_source_index);
  const sourceRow=retainedRow??(promptToken&&{source_index:promptToken.index,piece:promptToken.piece});
  const axesLabel=`layer ${state.attention_layer??0} · head ${state.attention_head??0}`;
  const generatorTarget=generation.data?.positions[state.attention_target_index??-1];
  const similarityTarget=capturedPage?.positions[state.similarity_target_index??-1];
  const sourcePopulation=(isGeneration?analysis:retrieverAttention.result)?.source_indices??[];
  const populationView=hasAttention&&!attentionColumnMode&&(sourcePopulation.length>1||(isGeneration?state.attention_method_id:state.retrieval_attention_method)==='attention.whole_query');
  // The exact row the page shows: the method sentence's tooltip and the Details › Result statement.
  const methodProperties=activeDescriptor?.parameters_schema.properties as Record<string,unknown>|undefined;
  const methodLayers=((isGeneration?state.attention_parameters:state.retrieval_attention_parameters)?.layers??(methodProperties?.layers as {default?:number[]}|undefined)?.default) as number[]|undefined;
  const layersText=methodLayers?.length?(methodLayers.length===1?`layer ${methodLayers[0]}`:`layers ${methodLayers.length<=4?methodLayers.join(', '):`${methodLayers.length} selected`}`):`layer ${state.attention_layer??0}`;
  const winnerCount=columns.similarity.data?.winning_query_indices.length??0;
  const pageContext:{testId:string;text:string}|undefined=attentionColumnMode?(generatorTarget?{testId:'column-context',text:`What attends to position ${generatorTarget.index} · ${generatorTarget.role==='image'?'image patch':displayPiece(generatorTarget.piece)} · ${axesLabel} · raw attention`}:undefined)
    :similarityColumnMode?similarityTarget&&{testId:'column-context',text:`Patch ${similarityTarget.index} · similarity to every query token · MaxSim winner for ${winnerCount} ${winnerCount===1?'token':'tokens'}`}
    :populationView?{testId:'source-population-context',text:`Captured source population · ${sourcePopulation.length} rows · ${layersText}`}
    :!isGeneration&&!retrieverAttention.enabled&&row.data&&capturedPage?{testId:'retrieval-winner',text:`Winner: position ${row.data.winner_index} · ${capturedPage.positions[row.data.winner_index].role} · dot product ${row.data.maximum.toFixed(6)}. ${capturedPage.positions[row.data.winner_index].bbox?'Gold outline marks the winning image patch.':'Non-page winner has no image box.'}`}:undefined;
  // "All" in the scrubber: a single-head method with a head mean, or that mean over every head (column mode is raw attention only).
  const headCount=(isRetrievalInvestigation?retrieverAttention.encoding?.query_head_count:generation.data?.query_head_count)??0;
  const headMode=attentionColumnMode?undefined:allHeadsState(isRetrievalInvestigation?retrieverMethod:state.attention_method_id,
    isRetrievalInvestigation?state.retrieval_attention_parameters:state.attention_parameters,headCount);
  const axesText=[!methodProperties?.layers&&`layer ${state.attention_layer??0}`,(!methodProperties||(!!methodProperties.head&&!methodProperties.heads))&&`head ${state.attention_head??0}`,headMode==='all'&&'all heads'].filter(Boolean).join(' · ');
  const rowSource:{testId:string;text:string}|undefined=isGeneration?imageSource(state.attention_source_index)?{testId:'prediction-alignment',text:`Image patch (position ${state.attention_source_index}) · full attention row · ${axesText}`}:sourceRow?{testId:'prediction-alignment',text:`${promptToken&&promptToken.role!=='query'?'Prompt token':'Question token'} “${displayPiece(sourceRow.piece)}” (position ${sourceRow.source_index})${promptToken?' · full attention row':''} · ${axesText}`}
    :prediction&&{testId:'prediction-alignment',text:`Step ${prediction.step} → “${displayPiece(prediction.predicted_piece)}” (position ${prediction.source_index} → ${prediction.predicted_index}) · ${axesText}`}
    :retrieverAttention.enabled?{testId:'retriever-attention-context',text:`${retrieverAttention.encoding?.pass_kind==='query'?'Query encoding':retrieverAttention.encoding?.pass_kind==='joint'?'Diagnostic joint page + question':'Document encoding'} · Source ${state.retrieval_source_index??0} · ${axesText}`}:undefined;
  const rowDetail=columnMode||populationView||!hasAttention?pageContext:rowSource;
  // Every retriever position, token or document patch, has its own row: a click makes it the source,
  // the same saved source position the panel's exact selects use.
  const chooseSource=(index:number)=>changeEvidence(s=>({...s,retrieval_source_index:index,retrieval_attention:true,attention_direction:'row',attention_target_index:null}));
  /** A generator token click shows its row; leaving a patch's column restores the method the patch click replaced. */
  function showRow(change:(s:ViewState)=>ViewState) {
    const restore=attentionColumnMode?rowMethod:undefined;
    changeEvidence(s=>{const next=change({...s,evidence_signal:'attention',attention_direction:'row'});return restore?switchMethod(next,restore,activeMethod.data,{preserveAxes:true}):next;});
    if(restore)setGenerationDetail(detailFor(restore.method_id));
    setRowMethod(undefined);
  }
  const chooseStep=(step:number)=>showRow(s=>({...s,attention_step:step,attention_source_index:null}));
  const chooseQuestion=(index:number)=>showRow(s=>({...s,attention_source_index:index}));
  /** With a full-matrix method on a full-attention capture, a patch click shows that patch's own row. */
  const patchSourceMode=!attentionColumnMode&&fullMatrixMethod(activeMethod.data);
  const choosePatchSource=(index:number)=>{if(fullAttention&&index<fullAttention.square_limit)showRow(s=>({...s,attention_source_index:index}));};
  /** A generator image patch has no captured row, so clicking it shows which rows attend to it (raw attention). */
  function showIncoming(index:number) {
    const current=state.attention_method_id??RAW_ATTENTION.method_id;
    if(!attentionColumnMode&&current!==RAW_ATTENTION.method_id)setRowMethod({method_id:current,variant_id:state.attention_variant_id??RAW_ATTENTION.variant_id});
    setSettings(s=>({...s,attention_grouping:'positions'}));
    changeEvidence(s=>({...((s.attention_method_id??RAW_ATTENTION.method_id)!==RAW_ATTENTION.method_id?switchMethod(s,RAW_ATTENTION,activeMethod.data):s),evidence_signal:'attention',attention_direction:'column',attention_target_index:index}));
    setGenerationDetail(detailFor(RAW_ATTENTION.method_id));
  }
  // MaxSim is symmetric to click: a query token shows its similarity to every patch, a patch to every query token.
  const showTokenSimilarity=(index:number)=>changeEvidence(s=>({...s,similarity_direction:'row',query_position:index}));
  const showPatchSimilarity=(index:number)=>changeEvidence(s=>({...s,similarity_direction:'column',similarity_target_index:index}));
  // A causal retriever source reaches only itself and earlier positions; population views state no reach.
  const reachSource=retrieverAttention.enabled&&retrieverAttention.encoding?.causal&&retrieverOnPage&&!populationView?retrieverAttention.row?.source_index:undefined;
  const reach:SourceReach|undefined=reachSource==null?undefined:(()=>{
    const mask=retrieverAttention.encoding!.attention_mask, last=Math.max(0,mask.lastIndexOf(1));
    const reaches=(index:number)=>index<=reachSource&&!!mask[index];
    const patches=retrieverPositions.filter(p=>p.bbox);
    return {source:reachSource,reaches,seen:patches.filter(p=>reaches(p.index)).length,total:patches.length,onWholePage:last!==reachSource?()=>chooseSource(last):undefined};
  })();
  const documentEncoding=retrieverAttention.enabled&&retrieverAttention.encoding?.pass_kind!=='query';
  const patchSources=documentEncoding&&retrieverOnPage?(retrieverAttention.positions??[]).filter(p=>p.bbox&&p.eligible).map(p=>({index:p.index,bbox:p.bbox!})):undefined;
  const opacity=settings.attention_opacity??.65;
  // MaxSim maxima are stored for eligible query positions only, in order.
  const eligibleRank=new Map((capture.data?.query_positions??[]).filter(p=>p.eligible).map((p,i)=>[p.index,i]));
  // Token heat shares the page's scale, except under image contrast, where tokens use their own maximum.
  const heatScale=(values:(number|null|undefined)[],full?:number)=>{
    const maximum=scaleMode==='auto'?Math.max(...values.map(v=>Math.abs(v??0)),1e-8):scaleMode==='causal'&&full!==undefined?full:scaleMax;
    return {scaleMax:maximum,opacity,scale:`Token scale 0 → ${attentionWeightLabel(maximum)}${scaleMode==='auto'?' (token maximum)':' (shared with page)'}.`};
  };
  const tokenHeat=(tokens:{index:number}[],values?:(number|null|undefined)[],full?:number)=>({values,...heatScale(tokens.map(t=>values?.[t.index]),full)});
  const questionTokens=generation.data?.positions.filter(p=>p.role==='query')??[];
  const sourceRows=generation.data?.source_rows??[];
  const questionSources=new Set(sourceRows.map(r=>r.source_index));
  const answerTokens=(generation.data?.steps??[]).map(s=>({index:s.step,piece:s.predicted_piece,role:'generated',title:`Step ${s.step}: consumed ${s.consumed_piece} → predicts ${s.predicted_piece}`}));
  const generatorHeat=tokenHeat(questionTokens,attention.data?.values.map((v,i)=>i<attention.data!.target_count?v:undefined));
  const queryValues=retrieverAttention.row?.values;
  const queryHeat=tokenHeat(capture.data?.query_positions??[],queryValues);
  const jointQuestionTokens=retrieverAttention.encoding?.pass_kind==='joint'?(retrieverAttention.positions??[]).filter(p=>p.role==='query_text'):[];
  // The reference answer in a Page + question prompt is a case annotation: its own strip, never "prompt".
  const jointAnswerTokens=retrieverAttention.encoding?.pass_kind==='joint'?(retrieverAttention.positions??[]).filter(p=>p.role==='answer_text'):[];
  const promptTokens=(retrieverAttention.positions??[]).filter(p=>!p.bbox&&!(retrieverAttention.encoding?.pass_kind==='joint'&&(p.role==='query_text'||p.role==='answer_text')));
  const promptHeat=tokenHeat([...promptTokens,...jointQuestionTokens,...jointAnswerTokens],queryValues);
  // Column view over the generator: answer tokens are prediction rows; question tokens are retained question rows.
  const answerColumn=(generation.data?.steps??[]).map(s=>generatorColumn[s.source_index]);
  const generatorColumnScale=heatScale([...questionTokens.map(p=>generatorColumn[p.index]),...answerColumn],columns.generator.data?.maximum);
  const similarityColumn=columns.similarity.data?.similarities;
  const similarityScale=settings.retrieval_scale_mode==='fixed'?settings.retrieval_scale_max??1:Math.max(...(similarityColumn??[]).map(v=>Math.abs(v??0)),1e-8);
  // A click on the page chooses the patch under the pointer; elements are chosen from the element list in Details › Element.
  // Generator patches are not clickable in the Document-aware view, where the page shows element masses.
  const generatorPatches=isGeneration&&consumed&&(attentionColumnMode||activePerspective!=='document_aware')?generation.data!.positions.filter(p=>p.page_id===pageId&&p.bbox).map(p=>({index:p.index,bbox:p.bbox!})):undefined;
  const similarityPatches=!hasAttention&&capturedPage?capturedPage.positions.filter(p=>p.bbox&&p.eligible).map(p=>({index:p.index,bbox:p.bbox!})):undefined;
  const pickable=patchSources?.length?{positions:patchSources,onPick:chooseSource}
    :generatorPatches?.length?{positions:generatorPatches,onPick:patchSourceMode?choosePatchSource:showIncoming}
    :similarityPatches?.length?{positions:similarityPatches,onPick:showPatchSimilarity}:undefined;
  // The clicked patch is outlined; its incoming weights or similarities are on the tokens, not the page.
  const outlineRow=(target:number,count:number):Attention=>({capture_id:'',step:0,layer:state.attention_layer??0,head:state.attention_head??0,
    source_index:target,predicted_index:null,target_count:count,values:Array.from({length:count},()=>0),quantity:'post_softmax_attention_weight',
    normalization:'column: each source row\'s weight on the target',total_mass:0,image_mass:0,allocations:[],allocation_error:0,display_max:1});
  const columnOverlay=attentionColumnMode&&generation.data&&consumed&&state.attention_target_index!=null
      ?{positions:generation.data.positions.filter(p=>p.page_id===pageId),row:outlineRow(state.attention_target_index,generation.data.positions.length),outlineOnly:true}
    :similarityColumnMode&&capturedPage&&state.similarity_target_index!=null
      ?{positions:capturedPage.positions.map((p,i)=>({...p,role:(p.bbox?'image':'special') as 'image'|'special',page_id:capturedPage.page_id,ownership:capturedPage.ownership[i]})),row:outlineRow(state.similarity_target_index,capturedPage.positions.length),outlineOnly:true}
    :undefined;
  // Every groupable method supplies its own allocations, including comparison rows.
  const documentAware=hasAttention&&!columnMode&&activePerspective==='document_aware';
  const regionRow=documentAware&&!documentGap?(retrieverAttention.enabled?(retrieverOnPage?retrieverAttention.row:undefined):isGeneration&&consumed?attention.data:undefined):undefined;
  const regionPositions=retrieverAttention.enabled?retrieverPositions:generation.data?.positions.filter(p=>p.page_id===pageId)??[];
  const pageMasses:Record<string,number>=regionRow?Object.fromEntries(regionRow.allocations.filter(a=>a.element_id&&a.page_id===pageId).map(a=>[a.element_id!,a.mass])):{};
  const contributing=new Set(regionRow&&state.selected_element_id?contributions(regionPositions,regionRow.values,regionRow.target_count,state.selected_element_id).map(p=>p.index):[]);
  const elementView=settings.attention_element_view??'elements';
  // The display toolbar chooses how this is drawn; Original keeps it available for the patch grid.
  const pageAttention=columnOverlay??(columnMode?undefined
    :documentAware?(regionRow&&elementView==='patches'?{positions:regionPositions,row:regionRow}
      :regionRow&&elementView==='selected_element'&&contributing.size?{positions:regionPositions.filter(p=>contributing.has(p.index)||(retrieverAttention.enabled&&p.index===regionRow.source_index)),row:regionRow}:undefined)
    :retrieverAttention.enabled&&retrieverAttention.row&&retrieverOnPage?{positions:retrieverPositions,row:retrieverAttention.row}
    :isGeneration&&generation.data&&attention.data&&consumed?{positions:generation.data.positions.filter(p=>p.page_id===pageId),row:attention.data}:undefined);
  const displayMode=attentionMode(settings,state.attention_overlay);
  // Element mass shares the patch palette with its own maximum across every page of the row.
  const regions:ElementRegions|undefined=regionRow?{masses:pageMasses,maximum:scaleMode==='auto'?elementMassMaximum(regionRow.allocations):scaleMax,mode:elementView==='elements'?displayMode:'original',opacity,quantity:elementQuantityLabel(regionRow.quantity),
    owned:new Set(regionPositions.flatMap(p=>Object.keys(p.ownership??{}).filter(b=>b.startsWith('element_')))),
    labelled:new Set(Object.entries(pageMasses).sort((a,b)=>b[1]-a[1]).slice(0,6).map(([id])=>id))}:undefined;
  function changeDisplayMode(next:AttentionMode) {
    changeView({...settings,attention_display_mode:next});
    if(state.attention_overlay===false)changeEvidence(s=>({...s,attention_overlay:true}));
  }
  function showAttentionPatches() {
    changeView({...settings,attention_element_view:'patches',attention_display_mode:displayMode==='original'?'overlay':displayMode,
      attention_opacity:settings.attention_opacity===0?.65:settings.attention_opacity});
    if(state.attention_overlay===false)changeEvidence(s=>({...s,attention_overlay:true}));
  }
  // Question tokens with a retained row are sources; older captures have answer rows only.
  const questionChoices=sourceRows.length||fullAttention?questionTokens.map(p=>({...p,eligible:questionSources.has(p.index)||!!promptSource(p.index)})):questionTokens;
  const generatorColumnNote=`Heat: weight each answer and question row gives the clicked patch (outlined). Raw attention at ${axesLabel}. ${generatorColumnScale.scale} Click a token to see what it attends to.${sourceRows.length?'':' This capture predates question-token rows, so only answer rows are shown.'}`;
  const generatorRowNote=(scale:string)=>`${sourceRows.length?'Click a question or answer token to see what it attends to. ':'This capture predates question-token rows; acquire a new generator capture to use question tokens as sources. '}Heat: attention from the selected source. ${scale} Image patches come before the question and have no row of their own: click a patch on the page to see which tokens attend to it.`;
  const queryNote=`Query encoding: click a query token to see what it attends to. The page is encoded separately. ${queryHeat.scale}`;
  const jointQuestionNote=`Page + question diagnostic: click a question token to explore its attention to the page. This reading is excluded from the retrieval score. ${promptHeat.scale}`;
  // Page-image patches, shown in Parts as one chip with the weight they receive together.
  const generatorImage=generation.data?.positions.filter(p=>p.role==='image')??[];
  const generatorImageWeight=attention.data?generatorImage.filter(p=>p.index<attention.data!.target_count).reduce((sum,p)=>sum+(attention.data!.values[p.index]??0),0):undefined;
  const retrieverImage=(retrieverAttention.positions??[]).filter(p=>p.bbox||p.role==='visual_patch');
  // The source line: question → answer for a generator, the reading's question and reference answer (or the query)
  // for a retriever, the query for MaxSim; the page chip ends the row. Template tokens are in the full prompt only.
  const retrieverSourceIndex=state.retrieval_source_index??0;
  const retrieverSource=(retrieverAttention.positions??[]).find(p=>p.index===retrieverSourceIndex);
  const sourceNotShown=retrieverAttention.enabled&&retrieverAttention.encoding?.pass_kind!=='query'&&retrieverSource&&retrieverSource.role!=='query_text'&&retrieverSource.role!=='answer_text';
  const templateNote=`${retrieverAttention.encoding?.pass_kind==='joint'?'Page + question diagnostic':'Page reading'}: click a patch on the page, or a template token in the full prompt, to see what it attends to. Template tokens are the model's fixed prompt text around the page and question. ${promptHeat.scale}`;
  const answerNoteText=`The reference answer from the case annotations, added to the prompt after the question. Click an answer token to see what it attends to; Last answer token in Details reads the position that predicts the final one. ${promptHeat.scale}`;
  const strips:ReactNode=isGeneration&&generation.data?(attentionColumnMode?<>
      <SourceSegment><TokenStrip label="Question tokens" caption="Question" tokens={questionChoices} values={generatorColumn} {...generatorColumnScale} selected={state.attention_target_index??undefined} onSelect={sourceRows.length?chooseQuestion:undefined}/></SourceSegment>
      <SourceSegment arrow><TokenStrip label="Answer tokens" caption="Answer" tokens={answerTokens} values={answerColumn} {...generatorColumnScale} onSelect={chooseStep}/></SourceSegment>
      <ImageChip patches={generatorImage.length}/>
    </>:<>
      <SourceSegment><TokenStrip label="Question tokens" caption="Question" tokens={questionChoices} {...generatorHeat}
        selected={state.attention_source_index??undefined} onSelect={sourceRows.length?chooseQuestion:undefined}/></SourceSegment>
      <SourceSegment arrow><TokenStrip label="Answer tokens" caption="Answer" tokens={answerTokens} selected={state.attention_source_index==null?state.attention_step??0:undefined} onSelect={chooseStep}/></SourceSegment>
      <ImageChip patches={generatorImage.length} weight={generatorImageWeight}/>
    </>)
    :retrieverAttention.enabled&&capture.data?retrieverAttention.encoding?.pass_kind==='query'
      ?<SourceSegment><TokenStrip label="Query tokens" caption="Query" tokens={capture.data.query_positions} {...queryHeat}
        selected={retrieverSourceIndex} onSelect={chooseSource}/></SourceSegment>
      :<>{!!jointQuestionTokens.length&&<SourceSegment><TokenStrip label="Question tokens" caption="Question" tokens={jointQuestionTokens} {...promptHeat}
        selected={retrieverSourceIndex} onSelect={chooseSource}/></SourceSegment>}
      {!!jointAnswerTokens.length&&<SourceSegment arrow><TokenStrip label="Answer tokens · reference answer" caption="Reference answer" tokens={jointAnswerTokens} {...promptHeat}
        selected={retrieverSourceIndex} onSelect={chooseSource}/></SourceSegment>}
      <ImageChip patches={retrieverImage.length} weight={queryValues&&retrieverImage.reduce((sum,p)=>sum+(queryValues[p.index]??0),0)}/>
      {sourceNotShown&&<SourceChip>Source · {retrieverSource.bbox?`page patch ${retrieverSource.index}`:<>template “{displayPiece(retrieverSource.piece)}” · position {retrieverSource.index}</>}</SourceChip>}</>
    :!hasAttention&&capture.data?(similarityColumnMode
      ?<SourceSegment><TokenStrip label="Query tokens" caption="Query" tokens={capture.data.query_positions} values={similarityColumn} signed scaleMax={similarityScale} opacity={settings.similarity_opacity??.65}
        marked={new Set(columns.similarity.data?.winning_query_indices??[])} onSelect={showTokenSimilarity}/></SourceSegment>
      :<SourceSegment><TokenStrip label="Query tokens" caption="Query" tokens={capture.data.query_positions.map(p=>({...p,title:`Position ${p.index} · ${p.piece} · ${p.role}${capturedPage?` · best match on this page ${capturedPage.query_maxima[eligibleRank.get(p.index)??-1]?.toFixed(6)??'—'}`:''}`}))}
        selected={state.query_position??0} onSelect={showTokenSimilarity}/></SourceSegment>)
    :null;
  // Notes for the source line's ⋯: how to choose a source, and the token scale.
  const sourceNotes:string[]=isGeneration&&generation.data?[attentionColumnMode?generatorColumnNote:generatorRowNote(generatorHeat.scale)]
    :retrieverAttention.enabled&&capture.data?(retrieverAttention.encoding?.pass_kind==='query'?[queryNote]:[...(jointQuestionTokens.length?[jointQuestionNote]:[]),...(jointAnswerTokens.length?[answerNoteText]:[]),templateNote])
    :!hasAttention&&capture.data?[similarityColumnMode?'Similarity of the clicked patch to every query token: signed MaxSim dot products, not attention. Gold borders mark tokens whose MaxSim winner is this patch. Click a token to see its similarity to every patch.'
      :'MaxSim similarity, not attention: click a token to see its signed similarity to every patch, or a patch on the page to see its similarity to every token.']:[];
  // Prompt view: the whole sequence in position order, as the model read it; PromptTokens folds each image run
  // into one chip. MaxSim encodes the query alone and compares it by similarity, so those modes keep Parts only.
  const byIndex=(a:{index:number},b:{index:number})=>a.index-b.index;
  const retrieverPart=(p:{role:string;bbox?:number[]|null}):PromptPart=>p.bbox||p.role==='visual_patch'?'image':p.role==='query_text'?'question':p.role==='answer_text'?'answer':'template';
  // Parts the shown reading leaves out stay in the Prompt view, dimmed where the Page + question reading puts them,
  // with a link to the reading that contains them: the question in a Page reading, the page in a Question reading.
  const encodings=capture.data?.attention_encodings??[];
  const readingOf=(kind:'document'|'joint',answered:boolean)=>encodings.filter(e=>e.pass_kind===kind&&!!e.answer_source===answered)
    .sort((a,b)=>Number(b.page_id===pageId)-Number(a.page_id===pageId))[0];
  const openReading=(next:NonNullable<typeof retrieverAttention.encoding>)=>changeEvidence(s=>({...s,retrieval_encoding_id:next.encoding_id,
    retrieval_source_index:defaultSource(next),retrieval_comparison_index:0,attention_target_index:null,attention_direction:'row',...(next.page_id?{page_id:next.page_id}:{})}));
  const jointPlain=readingOf('joint',false), jointAnswered=readingOf('joint',true), pageReading=readingOf('document',false);
  const textOf=(positions:{role:string;piece:string}[]|undefined,role:string)=>(positions??[]).filter(p=>p.role===role).map(p=>decodePiece(p.piece)).join('').replace(/[▁Ġ]/g,' ').replace(/Ċ/g,' ').trim();
  const ghostQuestion=textOf((jointPlain??jointAnswered)?.positions,'query_text')||textOf(capture.data?.query_positions,'query_text');
  const ghostAnswer:PromptGhost[]=jointAnswered?[{part:'answer',text:textOf(jointAnswered.positions,'answer_text'),
    reason:'This reading has no reference answer; Page + question with the reference answer adds it after the question.',
    action:{label:'Read with the reference answer',onClick:()=>openReading(jointAnswered)}}]:[];
  const shownPass=retrieverAttention.encoding?.pass_kind;
  const promptGhosts:PromptGhost[]=!retrieverAttention.enabled?[]
    :shownPass==='document'?[...(ghostQuestion?[{part:'question' as const,text:ghostQuestion,
      reason:jointPlain?'The Page reading encodes the page without the question; Page + question reads them together.':'The Page reading encodes the page without the question.',
      action:jointPlain&&{label:'Read with the question',onClick:()=>openReading(jointPlain)}}]:[]),...ghostAnswer]
    :shownPass==='joint'?(retrieverAttention.encoding?.answer_source?[]:ghostAnswer)
    :shownPass==='query'?[{part:'image',reason:'The Question reading encodes the question without the page.',
      action:(jointPlain??pageReading)&&{label:jointPlain?'Read with the page':'Show the Page reading',onClick:()=>openReading((jointPlain??pageReading)!)}}]
    :[];
  // Answer tokens beyond the captured prediction steps say why they cannot be chosen.
  const uncapturedAnswer=(piece:string)=>`Predicted ${piece} · not captured: attention rows are kept for the first ${generation.data!.steps.length} prediction steps only${fullAttention?' · this full-attention capture predates keeping every answer token':' · a capture with Save full attention matrices keeps every answer token'}`;
  const generatorPart=(role:string):PromptPart=>role==='image'?'image':role==='query'?'question':role==='generated'?'answer':'template';
  // Page-image chips in the prompt name their page by its number in the document.
  const pageLabel=(id?:string|null)=>id&&document.page_ids.includes(id)?`Page ${document.page_ids.indexOf(id)+1}`:undefined;
  const stepAt=new Map((generation.data?.steps??[]).map(s=>[s.predicted_index,s]));
  const generatorSequence:(Omit<PromptToken,'value'>&{step?:number})[]=generation.data?[
    ...generation.data.positions.map(p=>({index:p.index,piece:p.piece,role:p.role,part:generatorPart(p.role),step:stepAt.get(p.index)?.step,page:p.role==='image'?pageLabel(p.page_id):undefined})),
    ...generation.data.steps.filter(s=>!generation.data!.positions.some(p=>p.index===s.predicted_index))
      .map(s=>({index:s.predicted_index,piece:s.predicted_piece,role:'generated',part:'answer' as const,step:s.step})),
  ].map(t=>({...t,...(t.part==='answer'?{title:t.step!==undefined?`Step ${t.step}: predicts ${t.piece}`:uncapturedAnswer(t.piece),selectable:t.step!==undefined}
    :t.part==='question'?{selectable:sourceRows.length>0||!!fullAttention,eligible:!sourceRows.length&&!fullAttention||questionSources.has(t.index)||!!promptSource(t.index)}
    // Template tokens have a row only on a full-attention capture.
    :t.part==='template'&&promptSource(t.index)?{selectable:true,title:'Prompt token · full attention row'}:{selectable:false})})).sort(byIndex):[];
  const generatorText=generatorSequence.filter(t=>t.part!=='image');
  const generatorPromptHeat=tokenHeat(generatorText,attention.data?.values.map((v,i)=>i<attention.data!.target_count?v:undefined));
  // Columns are read from source rows only: question rows by position, answer tokens through the step that predicts them.
  const generatorPromptValue=(t:{index:number;part:PromptPart;step?:number})=>attentionColumnMode
    ?(t.part==='question'?generatorColumn[t.index]:t.part==='answer'&&t.step!==undefined?answerColumn[t.step]:undefined)
    :generatorPromptHeat.values?.[t.index];
  const chooseGeneratorToken=(t:PromptToken&{step?:number})=>t.part==='answer'?t.step!==undefined&&chooseStep(t.step):chooseQuestion(t.index);
  // The source line gives the prompt its view switch and options for the prompt's header.
  const promptLayout={open:!promptHidden,onOpen:(open:boolean)=>setPromptHidden(!open),height:promptHeight,onHeight:setPromptHeight};
  const retrieverPage=pageLabel(retrieverAttention.encoding?.page_id??pageId);
  const promptRow=isGeneration&&generation.data?(controls:ReactNode)=><PromptTokens label="Prompt" answerLabel="generated answer" controls={controls} {...promptLayout}
      tokens={generatorSequence.map(t=>({...t,value:generatorPromptValue(t)}))}
      {...(attentionColumnMode?generatorColumnScale:generatorPromptHeat)} onSelect={chooseGeneratorToken}
      selected={(t:PromptToken&{step?:number})=>attentionColumnMode?false:t.part==='answer'?state.attention_source_index==null&&t.step===(state.attention_step??0):t.index===state.attention_source_index}
      note={attentionColumnMode?generatorColumnNote:generatorRowNote(generatorPromptHeat.scale)}/>
    :retrieverAttention.enabled&&capture.data?retrieverAttention.encoding?.pass_kind==='query'
      ?(controls:ReactNode)=><PromptTokens label="Prompt" controls={controls} {...promptLayout} tokens={capture.data!.query_positions.map(p=>({...p,part:retrieverPart(p),value:queryValues?.[p.index]}))} {...queryHeat}
        selected={t=>t.index===(state.retrieval_source_index??0)} onSelect={t=>chooseSource(t.index)} ghosts={promptGhosts} note={queryNote}/>
      :(controls:ReactNode)=><PromptTokens label="Prompt" controls={controls} {...promptLayout} tokens={[...retrieverAttention.positions??[]].sort(byIndex).map(p=>{const part=retrieverPart(p);return {...p,part,value:queryValues?.[p.index],page:part==='image'?retrieverPage:undefined};})} {...promptHeat}
        selected={t=>t.index===(state.retrieval_source_index??0)} onSelect={t=>chooseSource(t.index)}
        last={retrieverAttention.encoding?.answer_source?lastAnswerToken(retrieverAttention.encoding)?.target.index:undefined} ghosts={promptGhosts}
        note={jointQuestionTokens.length?jointQuestionNote:`Page reading: click a patch on the page or a token to see what it attends to. ${promptHeat.scale}`}/>
    :null;
  const showPrompt=()=>{setFullPrompt(true);setPromptHidden(false);setImageFocus(false);};
  // P shows the full prompt, or hides it to its header when it is already shown.
  const togglePrompt=()=>{if(fullPrompt&&!promptHidden)setPromptHidden(true);else showPrompt();};
  const incomingColumn=columns.generator.data;
  // Incoming columns are not distributions, so the legend says so wherever the column is read.
  const incomingSummary=attentionColumnMode&&<span data-testid="incoming-summary">Incoming raw weights · {incomingColumn?`${incomingColumn.sources.length} captured source rows`:'source rows not loaded'} · values across sources do not sum to one</span>;
  // Exact totals of the shown row are an integrity record, kept with provenance; the legend states the shares.
  // Totals are named by the row's quantity: weights have a mass, envelopes and aggregates their own total.
  const rowTotalLabel=(quantity:string,domain:string)=>quantity==='post_softmax_attention_weight'?`${domain} mass`:quantity.includes('envelope')?'Head envelope total':`${quantityLabel(quantity)} total`;
  const rowTotalsFor=(ids:boolean)=>attentionColumnMode?undefined:isRetrievalInvestigation?retrieverAttention.row&&<p className="docattentionlens-row-summary" data-testid={ids?'retriever-attention-mass':undefined}>{rowTotalLabel(retrieverAttention.row.quantity,'Encoding-domain')} {scaleWeight(retrieverAttention.row.total_mass)} · image {scaleWeight(retrieverAttention.row.image_mass)} · non-image {scaleWeight(retrieverAttention.row.total_mass-retrieverAttention.row.image_mass)} · {retrieverAttention.row.target_count} encoding targets.</p>
    :attention.data&&<p className="docattentionlens-row-summary" data-testid={ids?'attention-mass':undefined}>{rowTotalLabel(attention.data.quantity,'Original-domain')} {scaleWeight(attention.data.total_mass)} · image {scaleWeight(attention.data.image_mass)} · non-image {scaleWeight(attention.data.total_mass-attention.data.image_mass)} · allocation error {(exactScale?attention.data.allocation_error.toExponential(2):roundedWeightLabel(attention.data.allocation_error))}.</p>;
  const rowTotals=rowTotalsFor(true);
  // Image contrast scales to the brightest patch, so the legend always states how much of the row reaches the page.
  const shownRow=columnMode?undefined:isGeneration?attention.data:retrieverAttention.enabled&&retrieverAttention.encoding?.pass_kind==='document'?retrieverAttention.row:undefined;
  const share=(mass:number)=>{const raw=shownRow&&shownRow.total_mass>0?mass/shownRow.total_mass:0;return sharePercent(Math.abs(raw)<1e-9?0:raw);};
  const imagePositions:{index:number;role:string;page_id?:string|null}[]=isGeneration?generation.data?.positions??[]:retrieverPositions;
  const pageImageMass=shownRow?imagePositions.filter(p=>p.role==='image'&&p.page_id===pageId&&p.index<shownRow.target_count).reduce((sum,p)=>sum+shownRow.values[p.index],0):0;
  // A row with no image mass at all (text-target blocks) is not "faint"; only a small positive share is.
  const lowImageMass=!!shownRow&&shownRow.image_mass>0&&shownRow.image_mass<.05*shownRow.total_mass;
  // What the shares are of: a weight row has a mass; rollouts, envelopes and aggregates have their own totals.
  const rowWhole=shownRow&&(shownRow.quantity==='post_softmax_attention_weight'?'row mass':shownRow.quantity.includes('rollout')?'rollout total':shownRow.quantity.includes('envelope')?'envelope total':shownRow.quantity==='whole_query_image_allocation'?'aggregate total':'block mass');
  const severalPages=isGeneration&&(generation.data?.context.pages.length??0)>1;
  // Details › Display states the share exactly as before; the legend says it in words.
  const imageShareText=shownRow&&`Image ${share(shownRow.image_mass)} of ${rowWhole}${severalPages?` · this page ${share(pageImageMass)}`:''}`;
  const legendShareText=shownRow&&`${severalPages?'Pages get':'Page gets'} ${share(shownRow.image_mass)} of ${shownRow.quantity==='post_softmax_attention_weight'?'attention':`the ${rowWhole}`}${severalPages?` · this page ${share(pageImageMass)}`:''}`;
  const lowMassText=lowImageMass&&scaleMode==='auto'?'Most attention is off the page; Image contrast magnifies small weights.':undefined;
  const imageShare=shownRow&&legendShareText&&<span data-testid="image-share" className={`docattentionlens-image-share ${lowImageMass?'is-low':''}`}
    title={`Share of the shown row's ${rowWhole} on page-image positions: ${attentionWeightLabel(shownRow.image_mass)} of ${attentionWeightLabel(shownRow.total_mass)}${severalPages?`; this page ${attentionWeightLabel(pageImageMass)}`:''}`}>{legendShareText}</span>;
  // Where the selected method's result is, when it is not on the page.
  const resultInEvidence=!hasAttention||attentionColumnMode?undefined
    :documentAware?(!documentGap&&(isGeneration?!legacyRow:retrieverMethod!=='attention.element_allocation')?'method result':undefined)
    :isGeneration?(analysis?.statistics?.length?'statistics':analysis?.comparisons?.length?'comparison maps':analysis?.table||analysis?.tables?.length?'summary table':analysis?.plots?.length?'chart':undefined)
    :retrieverAttention.result?.statistics?.length?'statistics':comparisonRows?.length?(retrieverAttention.encoding?.pass_kind==='query'?'comparison rows':'comparison maps')
    :retrieverAttention.result?.table||retrieverAttention.result?.tables?.length?'summary table':retrieverAttention.result?.plots?.length?'chart':undefined;
  const resultNotice=resultInEvidence&&<p role="status" key="result" className="docattentionlens-result-notice" data-testid="result-elsewhere">Result: {resultInEvidence} · <button className="docattentionlens-link-button" onClick={showEvidence}>Show in Details › Result</button></p>;
  // Notices sit on the page only when they explain a missing or faint heatmap.
  const columnRemote=isGeneration?columns.generator:columns.similarity;
  const columnTarget=similarityColumnMode?state.similarity_target_index:state.attention_target_index;
  const notices=(columnMode?[
    columnRemote.error&&<p role="alert" key="error">{columnRemote.error}</p>,
    columnTarget==null?<p role="status" key="target">Click a patch on the page to see what attends to it.</p>
      :!columnRemote.data&&!columnRemote.error&&<p role="status" key="loading"><Busy/>Reading what attends to the selected target…</p>,
  ]:isGeneration?[
    documentAware&&documentGap&&<p role="status" key="structure" data-testid="structure-gap">{documentGap}</p>,
    attention.error&&<p role="alert" key="error">{attention.error}</p>,
    analysisBody&&!legacyRow&&!analysis&&!methodAnalysis.error&&<p role="status" key="loading"><Busy/>Calculating selected method from saved evidence…</p>,
    !generation.data?(state.generation_capture_id&&!generation.error?<p role="status" key="loading"><Busy/>Loading generator capture…</p>:<p key="none">No generator capture selected.</p>):!consumed?<p role="status" key="page">The displayed document page was not supplied to this generator. It has no captured image attention.</p>:null,
    documentAware&&consumed&&regions&&displayMode==='original'&&<p role="status" key="regions">Original view: quantitative fills are hidden; boundaries remain. Choose Heatmap or Overlay in Details › Display, or press V.</p>,
    !documentAware&&consumed&&attention.data&&(displayMode==='original'?<p role="status" key="hidden">Original view: attention is hidden. Choose Heatmap or Overlay in Details › Display, or press V to see captured image weights.</p>:displayMode==='overlay'&&settings.attention_opacity===0?<p role="status" key="opacity">Attention opacity is zero. Increase Opacity in Display settings (⚙ in the legend) to see the overlay.</p>:imageMax===0?<p role="status" key="zero">This captured row has zero mapped image attention.</p>:scaleMode==='causal'&&imageMax/(attention.data.display_max||1)*(displayMode==='overlay'?settings.attention_opacity??.65:1)<.03?<p role="status" key="faint">Full-row scale makes image weights faint. Choose Image contrast in Display settings (⚙ in the legend) to inspect page patterns.</p>:null),
  ]:retrieverAttention.enabled?[
    retrieverAttention.error&&<p role="alert" key="error">{retrieverAttention.error}</p>,
    retrieverAttention.loading&&<p role="status" key="loading"><Busy/>Calculating selected method from saved evidence…</p>,
    // The element list states the reach; while it is not showing, this explains the unfilled elements.
    documentAware&&!documentGap&&!(ledgerLayout&&!imageFocus)&&!(detailsOpen&&detailsTab==='element')&&reach&&reach.seen<reach.total&&<p role="status" key="reach" data-testid="page-reach">Source {reach.source} sees {reach.seen} of {reach.total} page patches (causal encoder); later elements have no value.{reach.onWholePage&&<> <button className="docattentionlens-link-button" onClick={reach.onWholePage}>Use the last token (whole page)</button></>}</p>,
    documentAware&&documentGap?<p role="status" key="structure" data-testid="structure-gap">{documentGap}</p>
      :retrieverAttention.unavailable?<p role="status" key="unavailable" data-testid="method-unavailable">{retrieverAttention.unavailable}</p>
      :retrieverAttention.encoding?.pass_kind==='query'?<p role="status" key="query">Query encoding · weights target query positions, not this page.{resultNotice?'':' See Detailed evidence.'}</p>
      :!retrieverOnPage?<p role="status" key="page">This page has no captured internal attention. Select the captured document page or acquire a new capture.</p>
      :retrieverAttention.result&&!retrieverAttention.row&&!resultInEvidence?<p role="status" key="summary">The selected method reports a summary without a page overlay. See Detailed evidence.</p>:null,
  ]:[
    !state.retrieval_capture_id?<p key="none">No retrieval capture selected.</p>
      :capture.data&&!capturedPage?<p key="none">This page is outside the captured candidate set.</p>
      :!row.data&&!row.error&&!capture.error&&<p role="status" key="loading"><Busy/>Reading MaxSim similarities for this page…</p>,
  ]).filter(Boolean);
  if(resultNotice)notices.unshift(resultNotice);
  const selectedMethodName=hasAttention?(attentionColumnMode?'Raw incoming attention':isGeneration?activeDescriptor?.name??'Loading method…':retrieverAttention.method?.name??'Retriever attention'):'MaxSim similarity';
  const summaryRow=isGeneration?attention.data:retrieverAttention.enabled?retrieverAttention.row:undefined;
  const elementSummary=<ElementSummary row={summaryRow??undefined} positions={isGeneration?generation.data?.positions??[]:retrieverPositions} elements={elements.data?.items??[]}
    pageId={pageId} pageSupplied={isGeneration?!!consumed:retrieverOnPage} selectedId={state.selected_element_id} onSelect={choose} onPage={changePage}
    pageLabel={id=>`page ${document.page_ids.indexOf(id)+1}`} unavailable={documentAware?documentGap:undefined} provenance={provenanceLabel(structure,prepared.error)} documentAware={documentAware} support={support}
    methodName={selectedMethodName} onShowPatches={documentAware?showAttentionPatches:undefined} reach={reach}/>;
  const regionHint=documentAware&&regions&&elementView==='selected_element'&&!contributing.size&&<span data-testid="region-hint">{state.selected_element_id?'The selected element has no mapped patches':'Select an element to show its patches'}</span>;
  const receipt=researchReceipt({record,state,settings,generation:generation.data,capture:capture.data,role:isGeneration?'generation':'retrieval',signal:hasAttention?'attention':'similarity',methodName:selectedMethodName,scale:regions&&elementView==='elements'?regions.maximum:scaleMax,result:isGeneration?analysis:retrieverAttention.result,row:shownRow,column:attentionColumnMode?incomingColumn:undefined,quantity:isGeneration?activeMethod.data?.quantity:retrieverAttention.method?.quantity,units:isGeneration?activeMethod.data?.output.units:retrieverAttention.method?.output.units,similarityResult:similarityColumnMode?columns.similarity.data:row.data,displayMode:hasAttention?displayMode:state.similarity_overlay===false?'original':'signed overlay',pageQuantity:regionRow?(elementView==='elements'?elementQuantityLabel(regionRow.quantity):quantityLabel(regionRow.quantity)):undefined,groupedComparison:isGeneration?shownComparison?.label:retrieverAttention.result&&!retrieverAttention.result.row?selectedComparison(retrieverAttention.result.comparisons,state.retrieval_comparison_index)?.label:undefined,tokenScale:hasAttention?(isGeneration?attentionColumnMode?generatorColumnScale.scaleMax:generatorHeat.scaleMax:retrieverAttention.encoding?.pass_kind==='query'?queryHeat.scaleMax:promptHeat.scaleMax):similarityColumnMode?similarityScale:undefined});
  const sourceLabel=source.error??(source.data?`Element ${source.data.native_id} · page ${source.data.page_number}`:state.source_element_id?'Loading…':'Not selected');
  const targetLabel=target.error??(target.data?`Element ${target.data.native_id} · page ${target.data.page_number}`:state.target_element_id?'Loading…':'Not selected');
  const generationEvidence:GenerationDetail=generationView;
  const retrievalEvidence:RetrievalDetail=retrievalView;
  const evidenceView=isGeneration?generationEvidence:retrievalEvidence;
  const evidenceViews:[GenerationDetail|RetrievalDetail,string][]=isGeneration
    ?[...(!legacyRow?[['result','Method result'] as [GenerationDetail,string]]:[]),['tokens','Answer and prediction tokens'],['causal','Complete causal row']]
    :[...(retrieverAttention.enabled?[['result','Method result'],['encoding','Complete encoding row'],['allocations','Element summary']] as [RetrievalDetail,string][]:[]),['winners','Query winner ledger'],['contributions','Element contributions']];
  function chooseEvidence(view:GenerationDetail|RetrievalDetail) {if(isGeneration)setGenerationDetail(view as GenerationDetail);else setRetrievalDetail(view as RetrievalDetail);}
  const evidenceUnits=attentionColumnMode?'Incoming weights across captured sources':isGeneration?generationEvidence==='tokens'?'Selecting a token selects its prediction step':analysis?.units??'Post-softmax attention mass':retrievalEvidence==='winners'||retrievalEvidence==='contributions'?'Signed MaxSim score units':'Retriever attention weight';
  const modelLine=`${isGeneration?(generation.data?.provenance.model_id as string??'No generator capture'):(capture.data?.provenance.model_id as string??'No retrieval capture')} · capture ${(isGeneration?state.generation_capture_id:state.retrieval_capture_id)?.slice(-8)??'none'}`;
  // Table questions are checked against their accepted answers (shown beside this note); others are not evaluated here.
  const answerNote=generation.data&&`${generation.data.stop_reason==='eos'?'End token reached':'Token limit reached; answer may be incomplete'}${record.table_grounding?'':' · correctness has not been evaluated'}`;
  // Phones: Details and the pages open as bottom sheets over the bottom bar; Escape closes them.
  useEffect(()=>{
    if(wide||!sheet)return;
    const close=(e:KeyboardEvent)=>{if(e.key==='Escape'&&!e.defaultPrevented&&!window.document.querySelector('dialog[open],[role="dialog"][aria-modal="true"]'))setSheet(null);};
    window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close);
  },[wide,sheet]);
  useEffect(()=>{if(wide)setSheet(null);},[wide]);
  // One colour key for the shown token strip and the page: each range, what sets it and where its maximum is,
  // so a token and a patch of the same colour can be compared. It sits at the bottom of the right panel.
  const ramp=(onPage:boolean):Ramp=>hasAttention?{kind:onPage&&(displayMode==='heatmap'||displayMode==='blend')?displayMode:'amber',opacity,onPage,colormap:settings.attention_colormap??'inferno'}:{kind:'signed',opacity:settings.similarity_opacity??.65,onPage};
  type Strip=[{index:number;piece:string;role:string}[],(number|null|undefined)[]|undefined];
  const promptShown=fullPrompt&&!!promptRow;
  const tokenKey:{where:string;max:number;strips:Strip[];signed?:boolean}|undefined=imageFocus?undefined
    :isGeneration&&generation.data?(attentionColumnMode?{where:'Answer and question tokens',max:generatorColumnScale.scaleMax,strips:[[answerTokens,answerColumn],[questionTokens,generatorColumn]]}
      :promptShown?generatorPromptHeat.values&&{where:'Prompt',max:generatorPromptHeat.scaleMax,strips:[[generatorText,generatorPromptHeat.values]]}
      :generatorHeat.values&&{where:'Question tokens',max:generatorHeat.scaleMax,strips:[[questionTokens,generatorHeat.values]]})
    :retrieverAttention.enabled&&capture.data?queryValues&&(retrieverAttention.encoding?.pass_kind==='query'?{where:promptShown?'Prompt':'Query tokens',max:queryHeat.scaleMax,strips:[[capture.data.query_positions,queryValues]]}:{where:promptShown?'Prompt':jointAnswerTokens.length?'Question, answer & template tokens':jointQuestionTokens.length?'Question & template tokens':'Template tokens',max:promptHeat.scaleMax,strips:[[[...promptTokens,...jointQuestionTokens,...jointAnswerTokens],queryValues]]})
    :similarityColumnMode&&capture.data&&similarityColumn?{where:'Query tokens',max:similarityScale,strips:[[capture.data.query_positions,similarityColumn]],signed:true}:undefined;
  const tokenPeak=tokenKey&&(tokenKey.signed?settings.retrieval_scale_mode!=='fixed':scaleMode==='auto')?strongestToken(tokenKey.strips):undefined;
  const signedScale=(value:number)=>exactScale?value.toFixed(6):roundedWeightLabel(value);
  const tokenLine:ScaleLine|undefined=tokenKey&&{key:'tokens',where:tokenKey.where,ramp:{kind:tokenKey.signed?'signed':'amber',opacity:tokenKey.signed?settings.similarity_opacity??.65:opacity},rangeTestId:'token-scale',
    range:tokenKey.signed?`−${signedScale(tokenKey.max)} → ${signedScale(tokenKey.max)} · ${settings.retrieval_scale_mode==='fixed'?'fixed display scale':'column absolute maximum'}`
      :`0 → ${scaleWeight(tokenKey.max)} · ${scaleMode==='auto'?'token maximum':attentionColumnMode?'full column scale':'shared with page'}`,
    peak:tokenPeak&&<span className="docattentionlens-scale-token" title={`${tokenPeak.token.role==='generated'?'Step':'Position'} ${tokenPeak.token.index} · ${tokenKey.signed?'similarity':'attention weight'} ${attentionWeightLabel(tokenPeak.value)}`}>{displayPiece(tokenPeak.token.piece)}</span>};
  // Image contrast scales elements to the largest element mass, so that element is the page's peak.
  const peakAllocation=regionRow&&elementView==='elements'&&scaleMode==='auto'?regionRow.allocations.filter(a=>a.element_id&&a.mass>0).sort((a,b)=>b.mass-a.mass)[0]:undefined;
  const peakElement=peakAllocation&&elements.data?.items.find(e=>e.element_id===peakAllocation.element_id);
  // The legend states exact values; Details › Display states them as chosen there.
  const pageScaleLineIn=(weight:(value:number)=>string,signed:(value:number)=>string):ScaleLine|undefined=>documentAware?(regions&&regionRow?(elementView==='elements'
      ?{key:'page',where:'Page',ramp:ramp(true),rangeTestId:'element-scale',
        what:<span data-testid="element-key" title="Ownership-weighted sum of the selected method's position values in each annotated element.">{elementQuantityLabel(regionRow.quantity)}{displayMode==='original'?' (fills hidden)':''}</span>,
        range:`0 → ${weight(regions.maximum)} · ${scaleMode==='auto'?'element maximum':scaleMode==='fixed'?'fixed display scale; saturated values clipped visually':'full row display scale'}${scaleMode==='auto'&&isGeneration&&(generation.data?.context.pages.length??0)>1?' across supplied pages':''}`,
        peak:peakAllocation&&<button className="docattentionlens-link-button" title={`${attentionWeightLabel(peakAllocation.mass)} · select this element`} onClick={()=>void chooseById(peakAllocation.element_id!)}>{peakElement?`${peakElement.element_type} · element ${peakElement.native_id}`:`an element on page ${document.page_ids.indexOf(peakAllocation.page_id??'')+1}`}</button>}
      :{key:'page',where:'Page',ramp:ramp(true),rangeTestId:'display-scale',what:`${quantityLabel(regionRow.quantity)} · ${elementView==='patches'?'whole page including background':'selected element'}`,
        range:`0 → ${weight(scaleMax)} · ${scaleMode==='fixed'?'fixed patch scale':scaleMode==='causal'?'full row patch scale':'General patch scale · image contrast'}`}):undefined)
    :!columnMode&&(!hasAttention||attention.data||retrieverAttention.row)?{key:'page',where:retrieverAttention.enabled&&retrieverAttention.encoding?.pass_kind==='query'?'Query row':'Page',ramp:ramp(true),rangeTestId:'display-scale',
      what:retrieverAttention.enabled?`${quantityLabel(retrieverAttention.row?.quantity)}${sourcePatchExcluded?' · blue outline: source patch':''}`:isGeneration?quantityLabel(attention.data?.quantity):'MaxSim similarity · blue negative, lilac positive',
      range:`${hasAttention?'0':'−'+signed(scaleMax)} → ${hasAttention?weight(scaleMax):signed(scaleMax)} · ${scaleMode==='auto'?(retrieverAttention.enabled?(retrieverAttention.encoding?.pass_kind==='query'?'query row maximum · display only':(sourcePatchExcluded?'image maximum excluding the source patch · display only':'encoding image maximum · display only')):isGeneration?'image maximum across supplied pages · display only':'current row absolute maximum'):scaleMode==='causal'?(retrieverAttention.enabled?'row maximum over encoding targets':'row maximum over all causal targets'):'fixed display scale; saturated values clipped visually'}`}:undefined;
  // A range stretched from the page minimum states that minimum instead of 0.
  const floorValue=!hasAttention||settings.attention_range_floor!=='minimum'||displayMode==='original'?undefined
    :regions&&elementView==='elements'?rangeFloor(Object.values(regions.masses),'minimum')
    :pageAttention&&!('outlineOnly' in pageAttention&&pageAttention.outlineOnly)?rangeFloor(pageAttention.positions.filter(p=>p.bbox&&p.index<pageAttention.row.target_count).map(p=>pageAttention.row.values[p.index]),'minimum'):undefined;
  const pageLineIn=(weight:(value:number)=>string,signed:(value:number)=>string)=>{
    const line=pageScaleLineIn(weight,signed);
    return line&&floorValue!==undefined?{...line,range:line.range.replace(/^0 → /,`${weight(floorValue)} → `)+' · stretched from the page minimum'}:line;
  };
  const pageLine=pageLineIn(attentionWeightLabel,value=>value.toFixed(6));
  const pageLineShown=exactScale?pageLine:pageLineIn(roundedWeightLabel,roundedWeightLabel);
  const pageMax=regions&&elementView==='elements'?regions.maximum:scaleMax;
  // The page's scale beside the colour controls, with the value under the pointer: it covers nothing on the page.
  const elementFills=!!regions&&elementView==='elements'&&!pageAttention;
  const patchFills=!!pageAttention&&!('outlineOnly' in pageAttention&&pageAttention.outlineOnly);
  const hoverValue=pageHit?.kind==='patch'?(patchFills?pageAttention!.row.values[pageHit.index]:undefined):pageHit?.kind==='element'&&elementFills?regions!.masses[pageHit.id]:undefined;
  const pageKey=hasAttention&&pageLine&&displayMode!=='original'&&(patchFills||elementFills)&&<PageKey ramp={pageLine.ramp} floor={floorValue??0} maximum={pageMax}
    unit={elementFills?regions!.quantity??'Element attention mass':'Patch value'}
    hover={pageHit&&typeof hoverValue==='number'?{label:pageHit.kind==='patch'?`Patch ${pageHit.index}`:pageHit.label,value:hoverValue}:undefined}/>;
  const nonImageShareText=!pageLine?undefined:documentAware?regionRow&&`Non-image ${share(regionRow.total_mass-regionRow.image_mass)} · residuals in element summary`
    :shownRow&&`Non-image ${share(shownRow.total_mass-shownRow.image_mass)} · ${settings.attention_smooth?'smoothing is display only':'captured patches'}`;
  const nonImageShare=nonImageShareText&&<span data-testid="non-image-share">{nonImageShareText}</span>;
  // Every target group of the shown row, so the non-image share is broken down by role rather than left as one lump.
  const breakdownTargets=isGeneration?generation.data?.positions??[]:retrieverAttention.encoding?.positions??[];
  const breakdownGroups=pageLine&&shownRow?massBreakdown(shownRow.values.slice(0,shownRow.target_count),breakdownTargets,{pageId,severalPages,answerLabel:isGeneration?'Answer so far':'Reference answer'}):[];
  const scaleModeLabel=hasAttention?(scaleMode==='auto'?'Image contrast':scaleMode==='causal'?(retrieverAttention.enabled?'Full encoding row':'Full causal row'):'Fixed'):scaleMode==='fixed'?'Fixed':'Automatic per row';
  // Details › Display states the page line in full without the legend's test ids, so each value is stated once per place.
  const pageWhatPlain=documentAware&&regionRow&&elementView==='elements'?`${elementQuantityLabel(regionRow.quantity)}${displayMode==='original'?' (fills hidden)':''}`:pageLine?.what;
  const detailsLines=[tokenLine,pageLineShown&&{...pageLineShown,rangeTestId:undefined,what:pageWhatPlain}].filter((line):line is ScaleLine=>!!line);
  const scaleKey=<ScaleKey placement="panel" lines={detailsLines} mode={scaleModeLabel}
    note={tokenKey&&pageLine&&hasAttention&&pageMax>0?scaleAgreement(tokenKey.max,pageMax,scaleMode!=='auto',!exactScale):undefined} exact={exactScale} onExact={setExactScale}
    massShare={pageLine&&imageShareText&&shownRow&&shownRow.total_mass>0?shownRow.image_mass/shownRow.total_mass:undefined}
    breakdown={shownRow&&rowWhole&&breakdownGroups.length?{groups:breakdownGroups,total:shownRow.total_mass,whole:rowWhole,format:scaleWeight}:undefined}
    mass={pageLine&&(imageShareText||nonImageShareText)?<>{imageShareText&&<span>{imageShareText}{lowMassText?` · ${lowMassText}`:''}</span>}{nonImageShare}</>:undefined}/>;
  // The selection rules shared by the method picker on the canvas and the Details panel.
  const selection=useMethodSelection({role:isRetrievalInvestigation?'retrieval':'generation',generation:generation.data,retrieval:capture.data,retriever:retrieverAttention,
    prepared,state,update:changeEvidence,onMethod:selectMethod,settings,onView:changeView});
  const methodLabel=attentionColumnMode?'Raw incoming attention':selection.grouped?`${selection.input?attentionInputLabel(selection.input.method):'Attention'} → Element totals`:selectedMethodName;
  /** Records the Details layout in the saved display settings' existing fields: inspector open, dock open = Result. */
  const detailsLayout=(open:boolean,tab:DetailsTab)=>setSettings(s=>({...s,inspector_collapsed:!open,evidence_dock_open:open&&tab==='result',rail_tab:tab==='element'?'elements':'pages'}));
  // The three-column layout's drawer has no Method or Element tab: the method panel and the element inspector are their places.
  const drawerTab:DetailsTab=ledgerLayout&&(detailsTab==='method'||detailsTab==='element')?'result':detailsTab;
  function chooseDetailsTab(tab:DetailsTab) {
    if(ledgerLayout&&tab==='method'){openMethodPanel();return;}
    if(ledgerLayout&&tab==='element'){showInspector();return;}
    setDetailsTabState(tab);setPreferredTab(tab);
    if(tab!=='result'&&tab!=='element')setWideTable(false);
    if(wide&&detailsOpen)detailsLayout(true,tab);
  }
  /** Opens Details (a bottom sheet on phones), optionally on a tab, and moves focus into it. */
  function openDetails(tab?:DetailsTab,{focus=true,sheet:asSheet=true}:{focus?:boolean;sheet?:boolean}={}) {
    if(ledgerLayout&&tab==='method'){openMethodPanel();return;}
    if(ledgerLayout&&tab==='element'){showInspector(focus);return;}
    const next=tab??drawerTab;
    if(tab){setDetailsTabState(tab);setPreferredTab(tab);}
    if(!wide){if(asSheet){setSheet('details');if(focus)requestAnimationFrame(()=>inspectorPanel.current?.focus({preventScroll:true}));}return;}
    // The inspector's wide table does not carry over to the drawer that covers it.
    if(ledgerLayout&&!detailsOpen)setWideTable(false);
    setImageFocus(false);setMethodPanelOpen(false);detailsLayout(true,next);if(detailsWide)setDetailsPreference(true);
    if(focus)requestAnimationFrame(()=>inspectorPanel.current?.focus({preventScroll:true}));
  }
  function closeDetails() {
    setWideTable(false);
    if(!wide){setSheet(null);return;}
    detailsLayout(false,drawerTab);if(detailsWide)setDetailsPreference(false);
  }
  function toggleDetails() {if(detailsOpen)closeDetails();else openDetails(undefined,{focus:false});}
  /**
   * The three-column layout's element inspector: an open Details drawer closes, so the selected element shows beside
   * the page, and so does the method panel, as the Element tab replaced Method.
   */
  function showInspector(focus=false) {
    setImageFocus(false);setMethodPanelOpen(false);
    if(detailsOpen)closeDetails();
    if(focus)requestAnimationFrame(()=>elementPanel.current?.focus({preventScroll:true}));
  }
  /** The method panel under the method bar (three-column layout), with its search or the grouping choice focused. */
  function openMethodPanel(target:'search'|'grouping'='search') {
    setImageFocus(false);setMethodPanelOpen(true);
    requestAnimationFrame(()=>requestAnimationFrame(()=>window.document.querySelector<HTMLElement>(target==='grouping'?'#method-panel .docattentionlens-method-grouping [aria-checked="true"]':'#method-panel [data-testid="method-browser"] input[type="search"]')?.focus({preventScroll:true})));
  }
  function closeMethodPanel(refocus:boolean) {
    setMethodPanelOpen(false);
    if(refocus)requestAnimationFrame(()=>window.document.querySelector<HTMLElement>('.docattentionlens-method-step[data-step="method"],.docattentionlens-method-step[data-step="signal"]')?.focus());
  }
  /** A method change turns an open Details panel to the method's own output; a closed one stays closed. */
  function showResultIfOpen() {if(detailsOpen&&drawerTab!=='result'&&(ledgerLayout||detailsTab!=='method'))chooseDetailsTab('result');}
  /** Details › Method (or the method panel) with its search or the grouping choice focused: the method bar, M and the bottom bar. */
  function openMethods(target:'search'|'grouping'='search') {
    if(ledgerLayout){openMethodPanel(target);return;}
    openDetails('method',{focus:false});
    requestAnimationFrame(()=>window.document.querySelector<HTMLElement>(target==='grouping'?'.docattentionlens-method-grouping [aria-checked="true"]':'[data-testid="method-browser"] input[type="search"]')?.focus({preventScroll:true}));
  }
  const methodsOpen=ledgerLayout?methodPanelOpen:detailsOpen&&detailsTab==='method';
  // The method sentence: what the page shows in words and the method (opens the picker); the exact row is its tooltip.
  const generatorSourcePiece=sourceRow?displayPiece(sourceRow.piece):prediction?displayPiece(prediction.predicted_piece):undefined;
  const maxSimToken=capture.data?.query_positions.find(p=>p.index===(state.query_position??0));
  const sentenceLead=attentionColumnMode?(generatorTarget?`What attends to ${generatorTarget.role==='image'?`patch ${generatorTarget.index}`:`“${displayPiece(generatorTarget.piece)}”`}`:'What attends to a patch')
    :similarityColumnMode?(similarityTarget?`How patch ${similarityTarget.index} matches each query token`:'Patch similarity')
    :populationView?'What the captured sources attend to'
    :isGeneration?(imageSource(state.attention_source_index)?`What patch ${state.attention_source_index} attends to`
      :generatorSourcePiece!==undefined?`What “${generatorSourcePiece}” attends to`:undefined)
    :retrieverAttention.enabled?(retrieverSource?(retrieverSource.bbox?`What patch ${retrieverSource.index} attends to`:`What “${displayPiece(retrieverSource.piece)}” attends to`):undefined)
    :!hasAttention&&maxSimToken?`How “${displayPiece(maxSimToken.piece)}” matches the page`:undefined;
  const readingName=!retrieverAttention.encoding?undefined:retrieverAttention.encoding.pass_kind==='query'?'Question reading':retrieverAttention.encoding.pass_kind==='document'?'Page reading':retrieverAttention.encoding.answer_source?'Page + question + answer':'Page + question';
  // The method's maps (one per layer, head or condition) when it returns no single row: the page draws the chosen one.
  const retrieverMaps=retrieverAttention.result&&!retrieverAttention.result.row?retrieverAttention.result.comparisons:undefined;
  const generatorMaps=!legacyRow&&analysis&&!analysis.row?analysis.comparisons:undefined;
  const shownMaps=attentionColumnMode||!hasAttention?undefined
    :isGeneration?generatorMaps?.length?{labels:generatorMaps.map(c=>c.label),index:settings.attention_grouped_comparison_index??0,
      onChange:(index:number)=>changeView({...settings,attention_grouped_comparison_index:index})}:undefined
    :retrieverAttention.enabled&&retrieverMaps?.length?{labels:retrieverMaps.map(c=>c.label),index:state.retrieval_comparison_index??0,
      onChange:(index:number)=>changeEvidence(s=>({...s,retrieval_comparison_index:index}))}:undefined;
  const sentence=<MethodSentence testId={hasAttention&&!attentionColumnMode&&!populationView&&rowSource?'row-summary':undefined} lead={sentenceLead}
    method={!hasAttention?<strong className="docattentionlens-sentence-similarity">MaxSim similarity (not attention)</strong>
      // Phones (the bottom bar's Method opens Details › Method) and image focus have no method bar, so the sentence names the method.
      :!wide?<span className="docattentionlens-sentence-method-text" data-testid="method-choice" data-method-key={selection.primaryKey} data-perspective={selection.perspective}
        data-input-key={selection.grouped&&selection.input?methodKey(selection.input.method):undefined}>{methodLabel}</span>
      :imageFocus?<span className="docattentionlens-sentence-method-text">{methodLabel}</span>:undefined}
    extra={<>{shownMaps&&<ComparisonChoice labels={shownMaps.labels} index={shownMaps.index} onChange={shownMaps.onChange}/>}
      {!wide&&isRetrievalInvestigation&&!!capture.data?.attention_encodings?.length&&<button type="button" className="docattentionlens-sentence-chip" title="Choose the reading and the signal (attention or MaxSim) in Details › Result"
        aria-label={`${hasAttention?readingName??'Reading':'Signal: MaxSim similarity'} · choose in Details`} onClick={()=>openDetails('method')}>{hasAttention?readingName??'Reading':'Signal'}</button>}</>}
    detail={rowDetail&&<span data-testid={rowDetail.testId}>{rowDetail.text}</span>} detailText={rowDetail?.text}/>;
  const displayModeLabel:Record<AttentionMode,string>={original:'Original image',heatmap:'Heatmap',overlay:'Overlay',blend:'Blend'};
  const displaySummary=hasAttention?`${displayModeLabel[displayMode]} · ${scaleOptions(retrieverAttention.enabled).find(o=>o.id===scaleMode)?.label??scaleMode}`:'Document overlays';
  // One compact line on the page: ramp with floor and bound, the image share, ⓘ and the Display button. What the colour is
  // and what sets the bound are stated in full in Details › Display (and here for screen readers).
  const legend=<CanvasLegend ramp={pageKey||(pageLine&&<span className="docattentionlens-legend-ramp" style={rampStyle({...pageLine.ramp,onPage:true})} aria-hidden="true"/>)}
    bound={!pageKey&&pageLine?pageLine.range.split(' · ')[0]:undefined}
    range={pageLine&&<><span data-testid={pageLine.rangeTestId}>{pageLine.range}</span>{pageLine.where!=='Page'&&` · ${pageLine.where}`}{pageLine.what&&<> · {pageLine.what}</>}</>}
    share={imageShare} lowMass={lowMassText}
    notes={(regionHint||incomingSummary)&&<p className="docattentionlens-legend-note">{regionHint}{incomingSummary}</p>}
    settingsLabel={`Display settings · ${displaySummary}`} settingsText={hasAttention?displayModeLabel[displayMode]:'Display'}
    settingsOpen={detailsOpen&&drawerTab==='display'} onSettings={()=>detailsOpen&&drawerTab==='display'?closeDetails():openDetails('display',{focus:false})}
    disabled={hasAttention&&!(patchFills||!!regions||isGeneration&&!columnMode&&!!analysis?.comparisons?.length)}/>;
  const focusElement=selected?.display_bbox?()=>{setCanvasAction({id:Date.now(),box:selected?.display_bbox??null});setSheet(null);requestAnimationFrame(revealPage);}:undefined;
  const zoomPill=<ZoomPill fit={fitMode} zoom={settings.zoom??1}
    onZoomOut={()=>changeView({...settings,canvas_fit:'manual',zoom:Math.max(.25,(settings.zoom??1)/1.25)})} onZoomIn={()=>changeView({...settings,canvas_fit:'manual',zoom:Math.min(8,(settings.zoom??1)*1.25)})}
    onFit={next=>changeView({...settings,canvas_fit:next,zoom:1,center_x:null,center_y:null})} imageFocus={imageFocus} onImageFocus={toggleImageFocus} onFocusElement={focusElement}/>;
  // Layer and head under the page. Methods with a layer or head list keep those in their settings;
  // rollout methods show their layer interval.
  const scrubberAxes=!isRetrievalInvestigation?(generation.data?{layers:generation.data.layer_count,heads:generation.data.query_head_count}:undefined)
    :retrieverAttention.encoding&&capture.data?.attention_encodings?.length?{layers:retrieverAttention.encoding.layer_count,heads:retrieverAttention.encoding.query_head_count}:undefined;
  const layerShown=selection.rollout||!selection.properties?.layers;
  // The head mean over every head keeps the head cells, with All active.
  const headShown=!selection.rollout&&(!selection.properties||(!!selection.properties.head&&!selection.properties.heads)||headMode==='all');
  const rangeStart=typeof selection.inputParameters.start_layer==='number'?selection.inputParameters.start_layer:0;
  const chooseLayer=(value:number)=>selection.rollout?selection.changeLayerRange(selection.partialRollout?Math.min(rangeStart,value):0,value):selection.updateAttention(s=>({...s,attention_layer:value}));
  const chooseHead=(value:number)=>selection.updateAttention(s=>({...s,attention_head:value}));
  const axesLocked=selection.rollout&&!selection.selected?.available;
  // All switches to Mean across heads with every head selected explicitly, keeping the source, layer and the
  // remembered head; it is offered only when that method is available for this capture or encoding.
  const headMeanChoice=selection.all.find(p=>p.method.method_id===HEAD_MEAN_METHOD&&p.method.status!=='planned');
  const headReturnChoice=selection.all.find(p=>p.method.method_id===singleHeadReturn(headMethod)&&p.method.status!=='planned');
  const allHeadsOffered=!!scrubberAxes&&(headMode==='single'?!!headMeanChoice?.available:headMode==='all'&&!!headReturnChoice);
  /** Switches the shown method between a single head and the mean over all heads, in the same perspective and grouping. */
  function chooseAllHeads(on:boolean) {
    if(!allHeadsOffered||!scrubberAxes)return;
    if(on) {
      if(headMode!=='single'||!headMeanChoice)return;
      setHeadMethod(isRetrievalInvestigation?retrieverMethod:state.attention_method_id??RAW_ATTENTION.method_id);
      if(selection.grouped)selection.chooseInput(headMeanChoice.method);else selection.chooseMethod(headMeanChoice.method,true);
      const heads=everyHead(scrubberAxes.heads);
      selection.updateAttention(s=>isRetrievalInvestigation?{...s,retrieval_attention_parameters:{...s.retrieval_attention_parameters,heads}}:{...s,attention_parameters:{...s.attention_parameters,heads}});
    } else {
      if(headMode!=='all'||!headReturnChoice)return;
      if(selection.grouped)selection.chooseInput(headReturnChoice.method);else selection.chooseMethod(headReturnChoice.method,true);
      setHeadMethod(undefined);
    }
  }
  /** A numbered head from the keyboard or palette; while All is shown it returns to the single-head method first. */
  const pickHead=(value:number)=>{if(headMode==='all')chooseAllHeads(false);chooseHead(value);};
  // ---------- Element attention profile (Details › Element) ----------
  const [profileMetric,setProfileMetric]=useBrowserChoice<ProfileMetric>('docattentionlens-profile-metric','mass',isProfileMetric);
  // A retriever's page or diagnostic page + question encoding has its own profile of the selected source row.
  const retrieverProfileMethod=isRetrievalInvestigation?retrieverAttention.catalogue.data?.find(isElementProfile):undefined;
  const retrieverProfileGap=retrieverProfileMethod&&retrieverMappingGap(retrieverProfileMethod,retrieverAttention.encoding,retrieverAttention.page?.ownership,state.retrieval_source_index??0);
  const profileChoice=isRetrievalInvestigation?retrieverProfileMethod&&{method:retrieverProfileMethod,available:!retrieverProfileGap,reasons:retrieverProfileGap?[retrieverProfileGap]:[]}
    :prepared.data?.items.find(p=>isElementProfile(p.method)&&p.method.status!=='planned');
  // Image patches precede the question, so only sources after the last one see every element.
  const lastImage=Math.max(-1,...(isRetrievalInvestigation?(retrieverAttention.positions??[]).filter(p=>p.role==='visual_patch').map(p=>p.index)
    :(generation.data?.positions??[]).filter(p=>p.role==='image').map(p=>p.index)));
  const retrieverProfilePosition=isRetrievalInvestigation?(retrieverAttention.positions??[]).find(p=>p.index===(state.retrieval_source_index??0)):undefined;
  const profileSource=isRetrievalInvestigation?(retrieverProfilePosition&&retrieverProfilePosition.index>lastImage?{label:`${retrieverProfilePosition.role==='answer_text'?'Answer token':retrieverProfilePosition.role==='query_text'?'Question token':'Prompt token'} “${displayPiece(retrieverProfilePosition.piece??'')}” (position ${retrieverProfilePosition.index})`,sourceIndex:retrieverProfilePosition.index}:undefined)
    :state.attention_source_index==null?(prediction&&{label:`Step ${prediction.step} → “${displayPiece(prediction.predicted_piece)}”`,step:prediction.step})
    :sourceRow&&sourceRow.source_index>lastImage?{label:`${promptToken&&promptToken.role!=='query'?'Prompt token':'Question token'} “${displayPiece(sourceRow.piece)}” (position ${sourceRow.source_index})`,sourceIndex:sourceRow.source_index}:undefined;
  const profileCaptureId=isRetrievalInvestigation?capture.data?.capture_id:generation.data?.capture_id;
  const profileShown=(isGeneration?!!generation.data:retrieverAttention.enabled&&retrieverAttention.encoding?.pass_kind!=='query'&&!!capture.data)&&!tableCase&&!attentionColumnMode&&!!selected&&!!profileChoice;
  const profileBody=!isRetrievalInvestigation&&profileShown&&profileChoice.available&&profileSource?{generation_capture_id:generation.data!.capture_id,method_id:profileChoice.method.method_id,variant_id:profileChoice.method.variant_id,
    parameters:'step' in profileSource?{step:profileSource.step}:{},...('sourceIndex' in profileSource?{source_index:profileSource.sourceIndex}:{})}:undefined;
  const profileRetrieverBody=isRetrievalInvestigation&&profileShown&&profileChoice.available&&profileSource&&retrieverProfilePosition&&retrieverAttention.encoding
    ?{encoding_id:retrieverAttention.encoding.encoding_id,source_index:retrieverProfilePosition.index,method_id:profileChoice.method.method_id,parameters:{}}:undefined;
  const profileKey=JSON.stringify([profileCaptureId??null,profileBody??null,profileRetrieverBody??null]);
  const profileRequest=useRemote(useCallback(async()=>profileRetrieverBody&&profileCaptureId?{key:profileKey,result:{
      // Retriever results are recalculated from the saved capture, never stored; this names the exported selection.
      analysis_id:`retrieval ${profileCaptureId} · ${profileRetrieverBody.encoding_id} · source ${profileRetrieverBody.source_index}`,
      ...unwrap(await api.POST('/api/v1/retrieval/captures/{capture_id}/attention',{params:{path:{capture_id:profileCaptureId}},body:profileRetrieverBody}))}}
    :profileBody?{key:profileKey,result:unwrap(await api.POST('/api/v1/methods/analyses',{body:profileBody}))}:undefined,
  // The serialized request is the whole selection: the capture and the source, never the layer, head or element.
  // eslint-disable-next-line react-hooks/exhaustive-deps
    [profileKey]));
  const profileResult=profileRequest.data?.key===profileKey?profileRequest.data.result:undefined;
  const profileStatus:ProfileStatus=!profileChoice?{kind:'unavailable',reason:'Loading method availability…'}
    :!profileChoice.available?{kind:'unavailable',reason:profileChoice.reasons.join(' ')}
    :!profileSource?{kind:'unavailable',reason:'This source comes before the page image and cannot attend to its elements. Choose an answer step or a question token.'}
    :profileRequest.error?{kind:'error',message:profileRequest.error}
    :profileResult?.element_profile?{kind:'ready',profile:profileResult.element_profile,analysisId:profileResult.analysis_id,notices:profileResult.notices??[]}:{kind:'loading'};
  const profileElement=profileStatus.kind==='ready'?profileStatus.profile.elements.find(e=>e.element_id===selected?.element_id):undefined;
  // The scrubber's layer bars show the selected element's layer means in the profile's chosen value.
  const profileBars=profileShown&&profileStatus.kind==='ready'&&profileElement?(()=>{
    const means=layerMeans(profileStatus.profile,profileElement,profileMetric),top=Math.max(...means.map(m=>m??0),1e-12);
    return means.map(m=>m===null?Number.NaN:m/top);})():undefined;
  const profileLinked=!!scrubberAxes&&layerShown&&headShown&&!axesLocked&&!selection.rollout;
  /** Details › Element at Heads and layers (or the note saying what it needs), from the method browser and the palette. */
  function openHeadsAndLayers() {
    openDetails('element',{focus:false});
    requestAnimationFrame(()=>requestAnimationFrame(()=>(ledgerLayout?elementPanel:inspectorPanel).current?.querySelector('[data-testid=element-profile],[data-testid=element-profile-hint]')?.scrollIntoView({block:'start'})));
  }
  /** A profile cell on the page: the scrubber's layer and head, or (when the page method has its own axes) element aggregation there. */
  function showProfileCell(layer:number,head:number) {
    if(!profileLinked){const raw=selection.elementAnalysis;if(!raw?.available)return;selection.chooseMethod(raw.method,true,false,'document_aware');}
    else if(headMode==='all')chooseAllHeads(false);
    selection.updateAttention(s=>({...s,attention_layer:layer,attention_head:head}));
  }
  const scrubber=scrubberAxes&&<LayerScrubber layers={scrubberAxes.layers} heads={scrubberAxes.heads} layer={state.attention_layer??0} head={state.attention_head??0}
    onLayer={chooseLayer} onHead={chooseHead} showLayer={layerShown} showHead={headShown}
    bars={profileBars} barsTitle={profileElement?`Bars: layer mean of ${profileElement.label} (${PROFILE_METRICS.find(m=>m.id===profileMetric)?.label.toLowerCase()}) across heads · Details › Element`:undefined}
    allHeads={allHeadsOffered?{active:headMode==='all',onChange:chooseAllHeads}:undefined}
    range={selection.rollout?{from:selection.inputParameters.start_layer??0,to:state.attention_layer??0,fixedStart:!selection.partialRollout,onChooseRange:selection.chooseLayerRange,onChange:selection.changeLayerRange}:undefined}
    disabled={axesLocked}/>;
  const layerAxis=scrubberAxes&&layerShown&&!axesLocked?scrubberAxes.layers:undefined;
  const headAxis=scrubberAxes&&headShown&&!axesLocked?scrubberAxes.heads:undefined;
  const stepAxis=(value:number,count:number|undefined,by:number,set:(v:number)=>void)=>{if(!count)return;const from=Math.max(0,Math.min(count-1,value)),next=Math.max(0,Math.min(count-1,from+by));if(next!==from)set(next);};
  // `,` and `.` step through the source line's tokens; PageUp and PageDown through the pages in filmstrip order.
  const sourceSteps:{key:string;select:()=>void}[]=isGeneration&&generation.data?[
      ...(sourceRows.length?questionChoices.filter(t=>(t as {eligible?:boolean}).eligible!==false).map(t=>({key:`q${t.index}`,select:()=>chooseQuestion(t.index)})):[]),
      ...generation.data.steps.map(st=>({key:`s${st.step}`,select:()=>chooseStep(st.step)}))]
    :retrieverAttention.enabled&&capture.data?(retrieverAttention.encoding?.pass_kind==='query'?capture.data.query_positions
      :promptShown?[...(retrieverAttention.positions??[])].filter(p=>!p.bbox).sort(byIndex):[...jointQuestionTokens,...jointAnswerTokens]).filter(p=>p.eligible!==false).map(p=>({key:`r${p.index}`,select:()=>chooseSource(p.index)}))
    :!hasAttention&&capture.data?capture.data.query_positions.filter(p=>p.eligible!==false).map(p=>({key:`m${p.index}`,select:()=>showTokenSimilarity(p.index)})):[];
  const currentSourceKey=isGeneration?(attentionColumnMode?undefined:state.attention_source_index!=null?`q${state.attention_source_index}`:`s${state.attention_step??0}`)
    :retrieverAttention.enabled?`r${retrieverSourceIndex}`:similarityColumnMode?undefined:`m${state.query_position??0}`;
  function stepSource(by:number) {
    if(!sourceSteps.length)return;
    const at=sourceSteps.findIndex(s=>s.key===currentSourceKey);
    const next=at<0?(by>0?0:sourceSteps.length-1):Math.max(0,Math.min(sourceSteps.length-1,at+by));
    if(next!==at)sourceSteps[next].select();
  }
  // The pages to choose from, in filmstrip order: retrieval rank, or supplied then omitted pages.
  const pageNumberOf=(id:string)=>document.page_ids.indexOf(id)+1;
  const pageImage=(id:string)=>`/api/v1/pages/${encodeURIComponent(id)}/image`;
  // Each supplied page's share of the shown generator row, in the filmstrip and in Details › Run.
  const contextMass=(id:string)=>isGeneration&&attention.data&&generation.data?generation.data.positions.filter(p=>p.page_id===id&&p.index<attention.data!.target_count).reduce((total,p)=>total+attention.data!.values[p.index],0):undefined;
  const filmPages:FilmstripPage[]=!isGeneration&&capture.data?[...capture.data.pages].sort((a,b)=>b.score-a.score).map((p,i,all)=>({pageId:p.page_id,pageNumber:p.page_number,image:pageImage(p.page_id),
      state:`#${i+1}`,score:p.score,detail:`Rank ${i+1} of ${all.length} · signed MaxSim score ${p.score.toFixed(6)}`}))
    :isGeneration&&generation.data?[...generation.data.context.pages.map((p,i,all)=>({pageId:p.page_id,pageNumber:p.page_number,image:p.preview_url,state:'Supplied',mass:contextMass(p.page_id),
        detail:`Supplied page ${i+1} of ${all.length} · model input ${p.resized_width} × ${p.resized_height}`})),
      ...generation.data.context.omitted_page_ids.map(id=>({pageId:id,pageNumber:pageNumberOf(id),image:pageImage(id),state:'Omitted',omitted:true,
        detail:'Retrieved but omitted from the model input: no captured image attention'}))]
    :record.page_ids.map(id=>({pageId:id,pageNumber:pageNumberOf(id),image:pageImage(id),state:'',detail:'Case page'}));
  const pageOrder=filmPages.map(p=>p.pageId);
  const showFilmstrip=pageOrder.length>1;
  const massLabel=attention.data?(attention.data.quantity==='post_softmax_attention_weight'?'Original-domain attention mass':`${quantityLabel(attention.data.quantity)} total`):undefined;
  const evidenceLoaded=!!page.data&&!!elements.data&&(isGeneration?!!generation.data:!state.retrieval_capture_id||!!capture.data);
  const filmstrip=(orientation:'vertical'|'grid')=><Filmstrip pages={filmPages} current={pageId} orientation={orientation} massLabel={massLabel} loadImages={evidenceLoaded}
    onSelect={id=>{changePage(id);if(!wide)setSheet(null);}} label={isGeneration?'Generator context pages':'Retrieval candidate pages'}/>;
  function stepPage(by:number) {const at=pageOrder.indexOf(pageId);const next=pageOrder[Math.max(0,Math.min(pageOrder.length-1,(at<0?0:at)+by))];if(next&&next!==pageId)changePage(next);}
  // V cycles the display mode in the Colour mode order of Display settings.
  const MODES:AttentionMode[]=['original','heatmap','overlay','blend'];
  const cycleDisplayMode=()=>changeDisplayMode(MODES[(MODES.indexOf(displayMode)+1)%MODES.length]);
  useWorkspaceShortcuts({
    save:()=>{if(!saving&&name.trim())void save();},
    palette:()=>setPaletteOpen(true),
    help:()=>setShortcutsOpen(true),
    picker:hasAttention?()=>openMethods():undefined,
    details:toggleDetails, focus:toggleImageFocus, prompt:promptRow?togglePrompt:undefined,
    display:hasAttention?cycleDisplayMode:undefined,
    pagePrev:()=>stepPage(-1), pageNext:()=>stepPage(1),
    sourcePrev:()=>stepSource(-1), sourceNext:()=>stepSource(1),
    layerPrev:layerAxis?()=>stepAxis(state.attention_layer??0,layerAxis,-1,chooseLayer):undefined,
    layerNext:layerAxis?()=>stepAxis(state.attention_layer??0,layerAxis,1,chooseLayer):undefined,
    headPrev:headAxis?()=>stepAxis(state.attention_head??0,headAxis,-1,pickHead):undefined,
    headNext:headAxis?()=>stepAxis(state.attention_head??0,headAxis,1,pickHead):undefined,
    allHeads:allHeadsOffered&&!axesLocked?()=>chooseAllHeads(headMode!=='all'):undefined,
  },{blocked:paletteOpen||shortcutsOpen||!!drawer});
  const columnLabel=attentionColumnMode?{label:state.attention_target_index==null?'What attends to a patch · click one on the page':`What attends to ${generatorTarget?.role==='image'?'patch':'position'} ${state.attention_target_index}`,onClear:()=>showRow(s=>s),clearLabel:'Clear patch selection'}
    :similarityColumnMode?{label:state.similarity_target_index==null?'Patch similarity · click a patch on the page':`Similarity of patch ${state.similarity_target_index} to each query token`,onClear:()=>showTokenSimilarity(state.query_position??0),clearLabel:'Clear patch selection'}:undefined;
  const modelChip=<CaptureMenu isGeneration={isGeneration} generation={generation.data} retrieval={capture.data} captures={generationSession.records.data?.items} captureError={generationSession.records.error}
    onCapture={capture=>capture?generationSession.choose(capture):changeEvidence(s=>({...s,generation_capture_id:null}))} onAcquire={()=>openDrawer('acquire')} configureHref={`/attention-tool?case=${record.case_id}&configure=1`}/>;
  // A generation investigation can show its supporting retrieval scores; this returns to generator attention.
  const backToGeneration=!isRetrievalInvestigation&&!isGeneration?()=>changeEvidence(s=>({...s,evidence_signal:'attention'})):undefined;
  // The command palette: every choice the workspace offers, found by typing.
  const commands:Command[]=[
    ...selection.all.map((choice,i)=>({id:`method:${i}:${methodKey(choice.method)}`,label:choice.method.name,group:'Method',
      keywords:`method ${familyLabel(choice.method.family)} ${perspectiveLabel(choice.method.perspective)}`,hint:methodKey(choice.method)===selection.primaryKey?'Shown':perspectiveLabel(choice.method.perspective),
      disabled:!hasAttention?'MaxSim similarity has no attention methods; choose Retriever attention in Details › Result':choice.available&&choice.method.status!=='planned'?undefined:selection.reason(choice),
      run:()=>selection.choose(choice.method)})),
    // Every captured answer step (up to 64 on a full-attention capture), found by typing "step 40" or the token.
    ...(isGeneration&&generation.data?generation.data.steps.map(st=>({id:`step:${st.step}`,label:`Step ${st.step} · “${displayPiece(st.predicted_piece)}”`,group:'Answer token',keywords:'answer token prediction source',searchOnly:true,
      hint:!attentionColumnMode&&state.attention_source_index==null&&st.step===(state.attention_step??0)?'Shown':undefined,run:()=>chooseStep(st.step)})):[]),
    ...filmPages.map(p=>({id:`page:${p.pageId}`,label:`Page ${p.pageNumber}`,group:'Page',keywords:'go to page',hint:p.pageId===pageId?'Shown':p.state||undefined,run:()=>changePage(p.pageId)})),
    ...(layerAxis?Array.from({length:layerAxis},(_,i)=>({id:`layer:${i}`,label:`Layer ${i}`,group:'Layer',searchOnly:true,hint:i===(state.attention_layer??0)?'Shown':undefined,run:()=>chooseLayer(i)})):[]),
    ...(headAxis&&allHeadsOffered?[{id:'head:all',label:'All heads',group:'Head',keywords:'mean across every head',hint:headMode==='all'?'Shown':'A',run:()=>chooseAllHeads(true)}]:[]),
    ...(headAxis?Array.from({length:headAxis},(_,i)=>({id:`head:${i}`,label:`Head ${i}`,group:'Head',searchOnly:true,hint:headMode!=='all'&&i===(state.attention_head??0)?'Shown':undefined,run:()=>pickHead(i)})):[]),
    ...MODES.map(mode=>({id:`display:${mode}`,label:displayModeLabel[mode],group:'Display',keywords:'display mode colour',hint:hasAttention&&mode===displayMode?'Shown':undefined,
      disabled:hasAttention?undefined:'Display modes apply to attention; MaxSim uses its own overlay',run:()=>changeDisplayMode(mode)})),
    ...(!tableCase?[{id:'heads-layers',label:'Heads and layers for the selected element',group:'View',keywords:'element attention profile atlas strongest layer head model',run:openHeadsAndLayers}]:[]),
    {id:'display:settings',label:'Display settings',group:'Display',keywords:'colour map opacity scale smoothing overlays',run:()=>{setDisplayView('settings');openDetails('display');}},
    {id:'display:scale',label:'Colour scale and row mass',group:'Display',keywords:'scale range maximum mass breakdown totals rounded exact',run:()=>{setDisplayView('scale');openDetails('display');}},
    ...(ledgerLayout?DETAILS_TABS.filter(t=>DRAWER_TABS.includes(t.id)):DETAILS_TABS).map(t=>({id:`details:${t.id}`,label:`Details: ${t.label}`,group:'Details',keywords:t.title,run:()=>openDetails(t.id)})),
    ...(ledgerLayout?[{id:'method-panel',label:'Method: what it reads, its settings and the list',group:'Method',hint:'M',keywords:'method panel settings source reading grouping browse',run:()=>openMethods()}]:[]),
    ...(backToGeneration?[{id:'back-to-generation',label:'Back to generation attention',group:'View',keywords:'generator retrieval scores maxsim',run:backToGeneration}]:[]),
    ...(promptRow?[{id:'prompt',label:fullPrompt&&!promptHidden?'Hide full prompt':'Show full prompt',group:'View',hint:'P',keywords:'template tokens source line',run:togglePrompt}]:[]),
    {id:'focus',label:imageFocus?'Leave image focus':'Image focus',group:'View',hint:'F',run:toggleImageFocus},
    {id:'save',label:'Save investigation',group:'Action',hint:'Ctrl S',disabled:saving?'Saving…':!name.trim()?'Give the investigation a name first':undefined,run:()=>void save()},
    {id:'notes',label:'Rename and notes',group:'Action',run:()=>openDetails('notes')},
    {id:'provenance',label:'Provenance and receipt',group:'Action',run:()=>openDetails('run')},
    {id:'acquire',label:'Capture more evidence',group:'Action',run:()=>openDrawer('acquire')},
    {id:'studies',label:'Document studies',group:'Experiments',run:()=>openDrawer('research')},
    {id:'interventions',label:'Attention interventions',group:'Experiments',run:()=>openDrawer('interventions')},
    {id:'experiment',label:'Layout experiment',group:'Experiments',run:()=>openDrawer('experiment')},
    {id:'appearance',label:theme==='light'?'Appearance: Charcoal':'Appearance: White',group:'Appearance',keywords:'theme dark light',run:()=>setTheme(theme==='light'?'dark':'light')},
    {id:'shortcuts',label:'Keyboard shortcuts',group:'Help',hint:'?',run:()=>setShortcutsOpen(true)},
  ];
  // Heads and layers is always present in Details › Element (except table cases), saying why it is empty and what to do.
  const generatorCaptures=generationSession.records.data?.items??[];
  const profileIntro='Heads and layers shows which layers and heads attend to a selected element.';
  const profileHint:{text:string;action?:{label:string;run:()=>void}}|undefined=tableCase||profileShown?undefined
    :isRetrievalInvestigation&&retrieverAttention.enabled?(retrieverAttention.encoding?.pass_kind==='query'?{text:`${profileIntro} The query encoding has no page positions; choose the Page or Page + question reading.`}
      :!selected?{text:`${profileIntro} Select an element in the list ${ledgerLayout?'left of the page':'below'}.`}:undefined)
    :!isGeneration?(backToGeneration?{text:`${profileIntro} These are retrieval scores; return to the generator to use it.`,action:{label:'Back to generation attention',run:backToGeneration}}
      :generatorCaptures.length?{text:`${profileIntro} This view shows the retrieval model; open a generator capture of this question to use it.`,action:{label:'Open the generator capture',run:()=>generationSession.choose(generatorCaptures[0])}}
      :{text:`${profileIntro} This question has no generator capture yet; capture an answer to use it.`,action:{label:'Capture an answer',run:()=>openDrawer('acquire','attention')}})
    :attentionColumnMode?{text:`${profileIntro} Choose an answer or question token as the source to use it.`}
    :!selected?{text:`${profileIntro} Select an element in the list ${ledgerLayout?'left of the page':'below'}.`}:undefined;
  // ---------- Details tabs ----------
  const elementInspection=<>
    <ElementInspector selected={selected} retrieval={capture.data} generation={generation.data} attention={attention.data} isGeneration={isGeneration} sourceLabel={sourceLabel} targetLabel={targetLabel} sourceId={state.source_element_id} targetId={state.target_element_id} update={changeEvidence}
      profile={profileShown&&selected&&profileCaptureId&&<ElementProfile status={profileStatus} element={selected}
        suppliedPageIds={isRetrievalInvestigation?(retrieverAttention.encoding?.page_id?[retrieverAttention.encoding.page_id]:[]):generation.data!.context.selected_page_ids}
        source={{label:profileSource?.label??'Shown source',captureId:profileCaptureId,step:profileSource&&'step' in profileSource?profileSource.step:undefined,sourceIndex:profileSource&&'sourceIndex' in profileSource?profileSource.sourceIndex:undefined}}
        layer={state.attention_layer??0} head={headMode==='all'?-1:state.attention_head??0} linked={profileLinked} metric={profileMetric} onMetric={setProfileMetric}
        onShow={showProfileCell} onElement={id=>void chooseById(id)} colormap={settings.attention_colormap??'inferno'}/>}/>
    {profileHint&&<ProfileHint text={profileHint.text} action={profileHint.action}/>}
  </>;
  /** The element summary below the list or in the inspector; an open Details drawer gives way to the inspector first. */
  function showElementSummary() {
    if(ledgerLayout)showInspector();
    requestAnimationFrame(()=>requestAnimationFrame(()=>elementSummarySection.current?.scrollIntoView({block:'start'})));
  }
  const elementList=<section id="details-elements" className="docattentionlens-outline docattentionlens-details-section" aria-label="Page elements">
    <PageElements elements={ordered} selectedId={state.selected_element_id} onSelect={choose} row={!columnMode&&(isGeneration?!!consumed:retrieverOnPage)?summaryRow??undefined:undefined}
      positions={isGeneration?generation.data?.positions??[]:retrieverPositions} pageId={pageId} reach={reach} support={support}
      onDetails={hasAttention&&!columnMode?showElementSummary:undefined} markHref={document.ref.domain==='uploaded'?`/documents/${document.document_id}/elements`:undefined}/>
    {elements.error&&<Message error>{elements.error}</Message>}</section>;
  const elementSummaryPart=hasAttention&&!columnMode&&<section ref={elementSummarySection} className="docattentionlens-details-section docattentionlens-details-element-summary">{elementSummary}</section>;
  const elementTab=tableCase?<TableDetails tables={tables.data?.items} grounding={record.table_grounding} error={tables.error} settings={settings} onView={changeView}
    allocation={isGeneration?analysis?.table_allocation??undefined:retrieverAttention.result?.table_allocation??undefined} selectedCell={tableCell} onSelectCell={setTableCell}/>
    :<>{elementInspection}{elementList}{elementSummaryPart}</>;
  // The element inspector (three-column layout): what the Element tab shows beside the page, with its own wide table.
  const elementInspector=<aside ref={elementPanel} id="workspace-inspector" tabIndex={-1} className="docattentionlens-analysis-inspector docattentionlens-element-panel" aria-label="Element inspector"
    data-wide={wideTable} onKeyDown={e=>{if(e.key==='Escape'&&wideTable&&!e.defaultPrevented){e.preventDefault();setWideTable(false);}}}>
    <div className="docattentionlens-panel-title"><h2>Element</h2>
      {elementSummaryPart&&<button type="button" className="docattentionlens-quiet docattentionlens-details-wide" aria-pressed={wideTable}
        title={wideTable?'Return the element inspector to its column (Esc)':'Widen the element inspector over the page for the element summary'}
        onClick={()=>setWideTable(!wideTable)}>{wideTable?<Minimize size={14}/>:<Maximize size={14}/>}<span>{wideTable?'Close wide table':'Open as wide table'}</span></button>}
    </div>
    <div className="docattentionlens-details-body">{elementInspection}{elementSummaryPart}</div>
  </aside>;
  const evidenceContent=attentionColumnMode?<IncomingWeights column={incomingColumn} error={columns.generator.error}/>:isGeneration?generationEvidence==='result'?(analysis&&generation.data?<MethodResult onElement={id=>void chooseById(id)} receipt={receipt} result={analysis} generation={generation.data} pageId={pageId} scale={scaleMax} settings={settings} onView={changeView} onMode={changeDisplayMode} mode={displayMode} methodName={activeMethod.data?.name}/>:<p>{methodAnalysis.error??<>{analysisBody&&!analysis&&<Busy/>}Calculating selected method from saved evidence…</>}</p>)
    :generationEvidence==='tokens'?<AnswerTokens capture={generation.data} state={state} update={changeEvidence}/>
    :<GenerationDetails capture={generation.data} row={attention.data}/>
    :retrievalEvidence==='result'?(retrieverAttention.result?<RetrieverResult receipt={receipt} attention={retrieverAttention} state={state} update={changeEvidence} page={page.data} settings={settings} onView={changeView} onMode={changeDisplayMode} scale={scaleMax} mode={displayMode}/>:<p>{retrieverAttention.unavailable??retrieverAttention.error??<>{retrieverAttention.loading&&<Busy/>}Calculating selected method from saved evidence…</>}</p>)
    :retrievalEvidence==='allocations'?elementSummary
    :retrievalEvidence==='encoding'?<RetrieverRowDetails attention={retrieverAttention}/>
    :<><SimilarityControls capture={capture.data} state={state} settings={settings} update={changeEvidence} onView={changeView}/><RetrievalDetails capture={capture.data} row={row.data} state={state} update={changeEvidence} elements={elements.data?.items??[]} view={retrievalEvidence}/></>;
  // Result shows what came out: a summary of the setup (chosen in Details › Method), the grouped comparison to read, and the output.
  const editSetup=<button type="button" className="docattentionlens-link-button" onClick={()=>chooseDetailsTab('method')}>{ledgerLayout?'Change on the method bar':'Change in Method'}</button>;
  const resultTab=<>
    {rowDetail?<p className="docattentionlens-details-row"><span className="docattentionlens-details-kicker">Shown row</span> {rowDetail.text} · {editSetup}</p>
      :<p className="docattentionlens-details-row">The reading, source and method are chosen in {ledgerLayout?'the method panel':'Method'}. {editSetup}</p>}
    <GroupedComparison selection={selection} comparisons={isGeneration?analysis?.comparisons:retrieverAttention.result?.comparisons} state={state} settings={settings} onView={changeView}/>
    <section className="docattentionlens-details-output" aria-label="Method output">
      <div className="docattentionlens-evidence-head">
        {attentionColumnMode?<h3 className="docattentionlens-evidence-single">Incoming weights</h3>:<div className="docattentionlens-evidence-tabs" role="tablist" aria-label="Evidence view" onKeyDown={e=>{if(e.key!=='ArrowLeft'&&e.key!=='ArrowRight')return;e.preventDefault();e.stopPropagation();const at=evidenceViews.findIndex(([id])=>id===evidenceView);const next=evidenceViews[(at+(e.key==='ArrowRight'?1:-1)+evidenceViews.length)%evidenceViews.length][0];chooseEvidence(next);window.document.getElementById(`evidence-${next}`)?.focus();}}>
          {evidenceViews.map(([id,label])=><button key={id} role="tab" id={`evidence-${id}`} aria-selected={evidenceView===id} aria-controls="evidence-content" tabIndex={evidenceView===id?0:-1} onClick={()=>chooseEvidence(id)}>{label}</button>)}</div>}
        <small>{evidenceUnits}</small>
      </div>
      <div id="evidence-content" className="docattentionlens-evidence-content" role="tabpanel" aria-labelledby={attentionColumnMode?undefined:`evidence-${evidenceView}`}>{evidenceContent}</div>
    </section>
  </>;
  const displayTab=<DisplaySettingsTab settings={settings} hasAttention={hasAttention} encoding={retrieverAttention.enabled} onMode={changeDisplayMode} onView={changeView} structure={tableCase?'table':'layout'}
    view={displayView} onDisplayView={setDisplayView}
    readout={(scaleKey||rowTotals)&&<section className="docattentionlens-details-scale" aria-label="Scale and totals">
      {scaleKey}
      {rowTotals&&<div className="docattentionlens-details-totals"><h3>Shown row totals</h3>{rowTotals}</div>}
    </section>}
    inputView={isGeneration&&<label className="docattentionlens-display-select"><span className="docattentionlens-sr-only">Input view</span><select aria-label="Image view" disabled={!consumed||generation.data?.context.representation==='ocr'||generation.data?.context.representation==='structured'} value={inputView} onChange={e=>changeView({...settings,input_view:e.target.value as DisplaySettings['input_view']})}><option value="original">Original input</option><option value="consumed">Consumed input</option><option value="split">Compare inputs</option></select></label>}
    pageColouring={regionRow&&<DocumentAttentionView view={elementView} selected={!!selected} onChange={value=>changeView({...settings,attention_element_view:value})}/>}/>;
  const pageList=(id:string)=>({'data-page-id':id,'aria-pressed':pageId===id});
  const runTab=<>
    <section className="docattentionlens-details-section" aria-label="Capture"><h3>Model and capture</h3>
      <CaptureMenu inline isGeneration={isGeneration} generation={generation.data} retrieval={capture.data} captures={generationSession.records.data?.items} captureError={generationSession.records.error}
        onCapture={capture=>capture?generationSession.choose(capture):changeEvidence(s=>({...s,generation_capture_id:null}))} onAcquire={()=>openDrawer('acquire')} configureHref={`/attention-tool?case=${record.case_id}&configure=1`}/></section>
    <section className="docattentionlens-details-section docattentionlens-run-pages">
      {isGeneration?<ContextAttention capture={generation.data} row={attention.data} pageId={pageId} pageLabel={id=>`Page ${pageNumberOf(id)}`} onPage={changePage}/>:<>
      <h3>Candidate comparison</h3>
      <p>Signed MaxSim · selected candidates only.</p>
      {capture.data?<ol className="docattentionlens-page-list">{[...capture.data.pages].sort((a,b)=>b.score-a.score).map((p,i)=><li key={p.page_id}><button {...pageList(p.page_id)} onClick={()=>changePage(p.page_id)}><strong><small>#{i+1}</small> Page {p.page_number}</strong><span>{p.score.toFixed(6)}</span></button></li>)}</ol>
        :<p>No retrieval capture selected.</p>}</>}
    </section>
    <section className="docattentionlens-details-section docattentionlens-run-provenance" aria-label="Provenance and receipt">
      <ResearchReceipt receipt={receipt} caption={`Page ${(page.data?.page_number??0)} · ${record.query}`} canExportFigure={!!page.data&&!!((pageAttention&&!('outlineOnly' in pageAttention))||regions||(!hasAttention&&!columnMode&&row.data))}/>
      <RunProvenance record={record} generation={generation.data} capture={capture.data} isGeneration={isGeneration} isRetrievalInvestigation={isRetrievalInvestigation} onShowSimilarity={showSimilarity}/>
    </section>
  </>;
  const notesTab=<div className="docattentionlens-details-notes">
    {draftNotice}
    <label>Investigation name<input maxLength={200} value={name} onChange={e=>{setName(e.target.value);setDirty(true);}}/></label>
    <label>Notes<textarea aria-label="Notes" maxLength={20000} rows={12} value={notes} onChange={e=>{setNotes(e.target.value);setDirty(true);}}/></label>
    <div className="docattentionlens-actions"><button type="button" className="docattentionlens-button" title="Save the investigation (Ctrl+S)" disabled={saving||!name.trim()} onClick={()=>void save()}>Save name and notes</button><Link to="/artifacts/investigations">Saved investigations</Link></div>
  </div>;
  // The method bar: the shown method as labelled steps (reading, method, grouping); each opens Details › Method or Result.
  // In the three-column layout the reading and method steps open the method panel and Grouping is its own menu.
  const readings=isRetrievalInvestigation&&!!capture.data?.attention_encodings?.length;
  const groupingOffered=hasAttention&&!attentionColumnMode&&!tableCase&&selection.perspective!=='table_aware'&&!!selection.selected&&canGroupByElement(selection.selected.method);
  const where=ledgerLayout?'the method panel':'Details › Method';
  const panelStep=ledgerLayout?{expanded:methodPanelOpen}:{};
  const toggleMethodPanel=()=>methodPanelOpen?closeMethodPanel(false):openMethodPanel();
  const readingStep:MethodStep[]=readings?[{key:'reading',kicker:hasAttention?'Reading':'Signal',label:hasAttention?readingName??'Reading':'MaxSim similarity',
    title:`Choose the reading and the signal (attention or MaxSim) in ${where}`,onClick:ledgerLayout?toggleMethodPanel:()=>openDetails('method'),...panelStep}]:[];
  const elementTotals=selection.elementAnalysis;
  const groupingMenu:MethodStep['menu']=ledgerLayout?{label:'Grouping',items:[
    {label:'None',checked:!selection.grouped,title:"Show the method's weights on patches and tokens",onSelect:()=>{if(selection.grouped)selection.choosePerspective('general');}},
    {label:'Element totals',checked:selection.grouped,disabled:!elementTotals?.available,
      title:elementTotals?.available?"Add up the method's weights per document element":elementTotals?selection.reason(elementTotals):'Element totals are not offered for this capture',
      onSelect:()=>{if(!selection.grouped)selection.choosePerspective('document_aware');}}]}:undefined;
  const methodSteps:MethodStep[]=!hasAttention?readings?readingStep:[{key:'signal',kicker:'Signal',label:'MaxSim similarity (not attention)',title:`MaxSim similarity has no method choice; its query position is in ${where}`,onClick:ledgerLayout?toggleMethodPanel:()=>openDetails('method'),...panelStep}]
    :[...readingStep,
      {key:'method',kicker:selection.grouped||attentionColumnMode?'Attention':`${perspectiveLabel(selection.perspective)} method`,
        label:attentionColumnMode?'Raw incoming attention':selection.grouped?selection.input?attentionInputLabel(selection.input.method):'Attention':selectedMethodName,
        title:ledgerLayout?'The method: what it reads, its settings and the method list (M)':'Choose the method in Details › Method (M)',onClick:ledgerLayout?toggleMethodPanel:()=>openMethods(),...panelStep},
      ...(groupingOffered?[{key:'grouping',kicker:'Grouping',label:selection.grouped?'Element totals':'None',title:ledgerLayout?'Group the result into document elements, or not':'Group the result into document elements, or not, in Details › Method',onClick:()=>openMethods('grouping'),menu:groupingMenu}]:[])];
  // The arrows step through the ready methods of the list the method step is chosen from (the grouped inputs while grouped).
  const cycle=!hasAttention||attentionColumnMode?[]:(selection.grouped?selection.inputs:groupByFamily(filterChoices(selection.all,'',selection.perspective)).flatMap(g=>g.choices))
    .filter(p=>p.available&&p.method.status!=='planned');
  const cycleAt=cycle.findIndex(p=>methodKey(p.method)===(selection.grouped?selection.input&&methodKey(selection.input.method):selection.primaryKey));
  const cycleTo=(by:number)=>{const next=cycle[cycleAt+by];return next&&{title:`${by<0?'Previous':'Next'} method: ${next.method.name}`,go:()=>selection.choose(next.method,{asGroupingInput:selection.grouped})};};
  const prevMethod=cycleAt>=0?cycleTo(-1):undefined, nextMethod=cycleTo(1);
  // Details › Method holds everything that decides what is calculated: the signal, the reading and source, the
  // grouping, a table case's comparisons, whether the method can run, and the method with its settings.
  const signalChoice=isRetrievalInvestigation&&!!capture.data?.attention_encodings?.length&&<div className="docattentionlens-signal-choice"><label>Signal<select aria-label="Retrieval evidence type" value={state.retrieval_attention===false?'similarity':'attention'} onChange={e=>{if(e.target.value==='similarity')showSimilarity();else changeEvidence(s=>({...s,retrieval_attention:true}));}}><option value="attention">Retriever attention</option><option value="similarity">MaxSim similarity</option></select></label></div>;
  const methodSetup=<section className="docattentionlens-attention-panel docattentionlens-method-setup" aria-label="What the method reads">
    {signalChoice}
    {hasAttention?<AttentionSource selection={selection} generation={generation.data} retrieval={capture.data} retriever={retrieverAttention} state={state}
      onOpenEvidence={signal=>openDrawer('acquire',signal)} onChooseCapture={()=>chooseDetailsTab('run')} onSource={showRow} onShowPrompt={showPrompt} onAcquireAnswer={acquireAnswerReading}/>
      :<>{selection.isRetrieval&&!capture.data?.attention_encodings?.length&&<AttentionSource selection={selection} retrieval={capture.data} retriever={retrieverAttention} state={state}
          onOpenEvidence={signal=>openDrawer('acquire',signal)} onChooseCapture={()=>chooseDetailsTab('run')} onAcquireAnswer={acquireAnswerReading}/>}
        <SimilaritySource capture={capture.data} state={state} update={changeEvidence}/></>}
    <MethodReadiness selection={selection} isGeneration={isGeneration} generation={generation.data} preparedError={prepared.error}/>
  </section>;
  const methodTab=hasAttention?<div className="docattentionlens-method-tab">
    {methodSetup}
    {groupingOffered&&<MethodGrouping selection={selection}/>}
    {isGeneration&&tableCase&&<TableComparisons methodId={activeMethod.data?.method_id} related={tableRelated.data} error={tableRelated.error} useBaseline={tableBaseline} onBaseline={setTableBaseline} partner={partnerId} onPartner={setTablePartner}/>}
    {/* The current method's settings stay here whatever the list below previews. */}
    <div className="docattentionlens-attention-panel docattentionlens-method-current"><MethodSettings selection={selection} generation={generation.data} state={state}/></div>
    <MethodPicker open docked onClose={ledgerLayout?()=>closeMethodPanel(true):closeDetails} choices={selection.all} currentKey={selection.primaryKey}
      currentInputKey={selection.grouped&&selection.input?methodKey(selection.input.method):undefined} perspective={selection.perspective}
      onPerspective={selection.choosePerspective} perspectives={tableCase?['general','table_aware']:undefined}
      onSelect={(method,{asGroupingInput})=>selection.choose(method,{asGroupingInput})}
      catalogueHref={selection.detailUrl} readiness={selection.readiness} role={selection.role} encodingCount={selection.encodings.length}
      groupingInputs loading={!selection.catalogueReady} error={selection.catalogueError}
      // The element inspector shows Heads and layers beside the page in the three-column layout, so no pointer is needed there.
      notice={(isGeneration&&fullAttentionTwin)||(!tableCase&&!ledgerLayout)?<>{isGeneration&&fullAttentionTwin&&<><span data-testid="full-attention-twin">A full-attention capture of this run is saved: it adds rollout, flow, graphs, structure and image-patch sources.</span>
        <button type="button" className="docattentionlens-button" onClick={()=>generationSession.choose(fullAttentionTwin)}>Switch to it</button></>}
        {!tableCase&&!ledgerLayout&&<><span data-testid="heads-layers-pointer">Which layers and heads attend to one element: Heads and layers, in Details › Element.</span>
        <button type="button" className="docattentionlens-button" onClick={openHeadsAndLayers}>Open Heads and layers</button></>}</>:undefined}/>
  </div>
    :<div className="docattentionlens-method-tab">{methodSetup}<p className="docattentionlens-details-note">MaxSim similarity is not attention, so it has no method choice. Its ledgers and display are in <button type="button" className="docattentionlens-link-button" onClick={()=>chooseDetailsTab('result')}>Result</button>.</p></div>;
  const methodBar=<MethodBar steps={methodSteps} testId={hasAttention?'method-choice':undefined}
    data={hasAttention?{'method-key':selection.primaryKey,perspective:selection.perspective,'input-key':selection.grouped&&selection.input?methodKey(selection.input.method):undefined}:undefined}
    onPrev={prevMethod?.go} onNext={nextMethod?.go} prevTitle={prevMethod?.title} nextTitle={nextMethod?.title}
    onBrowse={hasAttention?()=>openMethods():undefined} browsing={methodsOpen}
    browseTitle={ledgerLayout?'All methods, their descriptions and settings (M)':undefined}
    panel={ledgerLayout?{open:methodPanelOpen,onClose:closeMethodPanel,label:'Method panel',content:methodTab,keepOpen:[sourceLine,scrubberBand]}:undefined}
    end={ledgerLayout?<DetailsToggle open={detailsOpen} onToggle={toggleDetails} controls="workspace-details" inBar/>:undefined}/>;
  const filmColumn=showFilmstrip&&wide&&!imageFocus?'72px':'0px';
  // The three-column layout always has the ledger and the right column (the element inspector, or the Details drawer over it).
  const ledgerShown=ledgerLayout&&!imageFocus;
  const rightColumn=ledgerShown||detailsOpen;
  const ledgerWidth=Math.min(LEDGER_MAX,Math.max(LEDGER_MIN,settings.rail_width??LEDGER_MIN));
  const ledgerColumn=ledgerShown?`${ledgerWidth}px`:'0px';
  const ledgerGap=ledgerShown?'6px':'0px';
  const detailsColumn=rightColumn&&wide?`${settings.inspector_width??360}px`:'0px';
  const detailsGap=rightColumn&&wide?'6px':'0px';
  return <div className="docattentionlens-analysis" data-sheet={sheet??'none'} data-details-open={detailsOpen} data-image-focus={imageFocus} data-layout={ledgerLayout?'columns':'panel'}
    style={{'--film-column':filmColumn,'--ledger-column':ledgerColumn,'--ledger-gap':ledgerGap,'--details-column':detailsColumn,'--details-gap':detailsGap,'--inspector-width':`${settings.inspector_width??360}px`} as CSSProperties}>
    <HeaderSlot><WorkspaceTopBar documentName={tableCase?`Table · ${(tables.data?.items.find(t=>t.table_id===record.table_grounding?.table_id)??tables.data?.items[0])?.title??document.filename}`:document.filename} question={record.query}
      details={{answer:isGeneration&&generation.data?generation.data.answer:undefined,answerNote:answerNote||undefined,referenceAnswer:record.reference_answer.trim(),wording:`${record.query_variant.replaceAll('_',' ')} wording · ${record.task_scope} scope`,modelLine,
        grounding:tableCase?'table_grounding':document.element_count>0?'document_layout':undefined,
        table:tableCase?{acceptedAnswers:record.table_grounding!.accepted_answers,check:answerCheck.data,source:document.filename}:undefined}}
      onBackToGeneration={backToGeneration}
      modelChip={modelChip} wide={wide} dirty={dirty} revision={investigation?.revision} saving={saving} canSave={!!name.trim()} onSave={()=>void save()}
      onNotes={()=>openDetails('notes')} onDisplay={()=>openDetails('display')} onStudies={()=>openDrawer('research')} onInterventions={()=>openDrawer('interventions')}
      onExperiment={()=>openDrawer('experiment')} onProvenance={()=>openDetails('run')} onPalette={()=>setPaletteOpen(true)} onShortcuts={()=>setShortcutsOpen(true)}/></HeaderSlot>
    <h1 className="docattentionlens-sr-only">Attention Tool</h1>
    <div ref={messages} className="docattentionlens-workspace-messages">
      {error&&<Message error>{error}{conflict&&<> Your edits remain here. <a href={window.location.href}>Reopen latest revision</a> after copying your notes.</>}</Message>}
      {!(detailsOpen&&drawerTab==='notes')&&draftNotice}
      {[capture.error,row.error,generation.error,attention.error,retrievalSession.error,generationSession.error].filter(Boolean).map((message,i)=><Message key={i} error>{message}</Message>)}
      {[retrievalSession.jobId,generationSession.jobId].filter(Boolean).map(id=><Message key={id}><Busy/>Evidence acquisition in progress. <Link to={`/artifacts/jobs/${id}`}>Job details</Link><button className="docattentionlens-button" onClick={()=>void api.POST('/api/v1/jobs/{job_id}/cancel',{params:{path:{job_id:id!}}})}>Cancel acquisition</button></Message>)}
    </div>
    <div id="analysis-task" role="region" aria-label={isRetrievalInvestigation?'Retrieval analysis':isGeneration?'Generation analysis':'Supporting retrieval scores'} className="docattentionlens-analysis-body">
      {showFilmstrip&&wide&&!imageFocus&&<div className="docattentionlens-film-column">{filmstrip('vertical')}</div>}
      {ledgerShown&&<aside className="docattentionlens-ledger-column" aria-label="Element ledger">{elementList}</aside>}
      {ledgerShown&&<div className="docattentionlens-ledger-resizer"><ResizeHandle label="Resize the element list" value={ledgerWidth} min={LEDGER_MIN} max={LEDGER_MAX} onChange={v=>changeView({...settings,rail_width:v})}/></div>}
      <section className="docattentionlens-analysis-document" aria-label="Document page">
        <div ref={sourceLine} className="docattentionlens-source-band" hidden={imageFocus}>
          <SourceLine column={columnLabel} prompt={promptRow??undefined} fullPrompt={fullPrompt} onFullPrompt={setFullPrompt} notes={sourceNotes} selectedKey={`${currentSourceKey}|${sourceSteps.length}|${retrieverAttention.encoding?.encoding_id??''}`}>{strips}</SourceLine>
          {wide&&methodBar}
        </div>
        <div ref={imageViews} className={`docattentionlens-image-views ${inputView==='split'?'is-split':''}`} data-fit={fitMode} data-page-id={pageId} style={page.data?{'--page-aspect':`${page.data.width} / ${page.data.height}`} as CSSProperties:undefined}>
          {(retrievalSession.jobId||generationSession.jobId)&&<BusySweep/>}
          <div className="docattentionlens-image-view" style={inputView==='consumed'&&consumed?{maxWidth:`${consumed.resized_width*2}px`}:undefined}>{inputView==='split'&&<span>Original document</span>}{canvas(inputView==='consumed')}</div>
          {inputView==='split'&&<div className="docattentionlens-image-view"><span>Consumed BF16 input · source coordinates</span>{canvas(true)}</div>}
          <CanvasInstruments topLeft={<>{sentence}{!!notices.length&&<div className="docattentionlens-canvas-notices" aria-live="polite">{notices}</div>}</>}
            topRight={ledgerLayout?undefined:<DetailsToggle open={detailsOpen} onToggle={toggleDetails} controls="workspace-details"/>} bottomLeft={legend} bottomRight={zoomPill}/>
        </div>
        <div ref={scrubberBand} className="docattentionlens-scrubber-band">{scrubber}</div>
      </section>
      {wide&&rightColumn&&<div className="docattentionlens-inspector-resizer"><ResizeHandle label={detailsOpen?'Resize Details':'Resize the element inspector'} value={settings.inspector_width??360} min={Math.min(300,saved?.display_settings?.inspector_width??300)} max={560} reverse onChange={v=>changeView({...settings,inspector_width:v})}/></div>}
      {ledgerShown&&!detailsOpen&&elementInspector}
      <DetailsPanel tab={drawerTab} onTab={chooseDetailsTab} label={isRetrievalInvestigation?'Retrieval panel':'Attention panel'} panelRef={inspectorPanel} hidden={!detailsOpen} labels={tableCase?{element:{label:'Table',title:"The table on this page, how it was located and the question's lookup path"}}:undefined}
        tabs={ledgerLayout?DRAWER_TABS:undefined} closeTitle={ledgerLayout?'Close Details and show the selected element (D)':undefined}
        content={{method:methodTab,element:elementTab,result:resultTab,display:displayTab,run:runTab,notes:notesTab}} wideTable={wideTable} onWideTable={setWideTable} onClose={closeDetails} sheet={!wide}/>
    </div>
    {!wide&&showFilmstrip&&sheet==='pages'&&<aside className="docattentionlens-pages-sheet" aria-label="Pages">
      <div className="docattentionlens-panel-title"><h2>Pages</h2><button type="button" className="docattentionlens-icon-button" aria-label="Close pages" title="Close pages (Esc)" onClick={()=>setSheet(null)}><X/></button></div>
      {filmstrip('grid')}</aside>}
    <nav ref={bottomBar} className="docattentionlens-bottom-bar" aria-label="Workspace views">
      <button type="button" className="docattentionlens-quiet" aria-pressed={!sheet} onClick={()=>{setSheet(null);revealPage();}}><FileText/><span>Page</span></button>
      <button type="button" className="docattentionlens-quiet" aria-pressed={sheet==='details'} aria-controls="workspace-details" onClick={()=>sheet==='details'?setSheet(null):openDetails()}><PanelRight/><span>Details</span></button>
      <button type="button" className="docattentionlens-quiet" aria-pressed={sheet==='details'&&detailsTab==='method'} aria-controls="workspace-details" disabled={!hasAttention} title={hasAttention?'Choose the analysis method in Details › Method':'MaxSim similarity has no method choice'} onClick={()=>sheet==='details'&&detailsTab==='method'?setSheet(null):openMethods()}><Sliders/><span>Method</span></button>
      {showFilmstrip&&<button type="button" className="docattentionlens-quiet" aria-pressed={sheet==='pages'} onClick={()=>setSheet(v=>v==='pages'?null:'pages')}><Files/><span>Pages</span></button>}
    </nav>
    {drawer==='acquire'&&<Drawer title="Open / acquire evidence" onClose={()=>setDrawer(null)}>
      {acquisitionSignal==='attention'?<GenerationPanel record={record} retrieval={capture.data} capture={generation.data} state={state} session={generationSession} update={changeEvidence}/>:<RetrievalPanel record={record} document={document} state={state} capture={capture.data} session={retrievalSession}/>}
    </Drawer>}
    <Drawer title="Layout experiment" open={drawer==='experiment'} onClose={()=>setDrawer(null)}><ExperimentPanel key={pageId} record={record} pageId={pageId} elements={elements.data?.items??[]}/></Drawer>
    {drawer==='interventions'&&<Drawer title="Attention interventions" wide onClose={()=>setDrawer(null)}><AttentionInterventions record={record} generation={generation.data} retrieval={capture.data} role={isGeneration?'generation':'retrieval'} pageId={pageId} layer={state.attention_layer??0} head={state.attention_head??0} sourceIndex={isGeneration?state.attention_source_index??generation.data?.steps.find(s=>s.step===(state.attention_step??0))?.source_index:retrieverAttention.encoding?.pass_kind==='query'?undefined:state.retrieval_source_index??0} targetIndex={state.attention_target_index}/></Drawer>}
    {drawer==='research'&&<Drawer title="Document studies" wide onClose={()=>setDrawer(null)}><DocumentStudies record={record} generation={generation.data} elements={elements.data?.items??[]} onElement={id=>void chooseById(id)}/></Drawer>}
    <CommandPalette open={paletteOpen} onClose={()=>setPaletteOpen(false)} commands={commands}/>
    <ShortcutSheet open={shortcutsOpen} onClose={()=>setShortcutsOpen(false)} groups={SHORTCUT_GROUPS}/>
  </div>;
}
