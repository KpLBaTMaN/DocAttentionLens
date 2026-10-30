/** One category vocabulary for Toolbox and the workspace method picker. */
export const FAMILY_LABELS: Record<string,string> = {
  direct_attention:'Attention · Direct inspection',
  aggregation:'Attention · Aggregation',
  multimodal_attention:'Attention · Modality blocks',
  attention_statistics:'Attention · Statistics and sinks',
  spatial_attention:'Attention · Spatial structure',
  head_and_representation:'Attention · Heads and patterns',
  connectivity:'Attention-derived · Propagation and flow',
  attention_graphs:'Attention · Raw edge graphs',
  document_structure:'Attention · Document structure',
  table_structure:'Attention · Table structure',
  attention_interventions:'Attention interventions · Controlled model runs',
  similarity_and_retrieval:'Retrieval · Similarity and scoring',
  output_attribution:'Output attribution',
  input_perturbation:'Input perturbation',
  perturbations_and_interventions:'Input and layout experiments',
};
export const familyOrder=Object.keys(FAMILY_LABELS);
