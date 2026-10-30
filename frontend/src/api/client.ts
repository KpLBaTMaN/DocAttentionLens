import createClient from 'openapi-fetch';
import type { components, paths } from './generated/schema';

export const api = createClient<paths>({ baseUrl: '' });
export type Method = components['schemas']['MethodDescriptor'];

export function errorMessage(error: unknown): string {
  if (typeof error === 'object' && error !== null && 'error' in error) {
    const detail = (error as { error: { message?: string } }).error;
    return detail.message ?? 'The request failed.';
  }
  return error instanceof Error ? error.message : 'The request failed.';
}

/** Gateway and availability failures carry no JSON body; name them instead of a bare "The request failed." */
const STATUS_MESSAGES: Record<number, string> = {
  502: 'The API is not reachable (HTTP 502). Check that the API container is running.',
  503: 'The service is temporarily unavailable (HTTP 503). Try again.',
  504: 'The server took too long to respond (HTTP 504). Try again.',
};

export function unwrap<T>(result: { data?: T; error?: unknown; response?: Response }): T {
  if (result.data !== undefined) return result.data;
  const status = result.response?.status;
  const hasMessage = typeof result.error === 'object' && result.error !== null && 'error' in result.error;
  throw new ApiError(!hasMessage && status ? {error: {message: STATUS_MESSAGES[status] ?? `The request failed (HTTP ${status}).`, code: `http_${status}`}} : result.error);
}

export class ApiError extends Error {
  code: string;
  detail: Record<string, unknown>;
  constructor(value: unknown) {
    super(errorMessage(value));
    this.detail = typeof value === 'object' && value !== null && 'error' in value ? (value as {error: Record<string, unknown>}).error : {};
    this.code = typeof this.detail.code === 'string' ? this.detail.code : 'request_failed';
  }
}

export type Document = components['schemas']['DocumentRecord'];
export type Page = components['schemas']['PageRecord'];
export type Element = components['schemas']['ElementRecord'];
export type Case = components['schemas']['CaseRecord'];
export type Investigation = components['schemas']['InvestigationRecord'];
export type DisplaySettings = components['schemas']['DisplaySettings'];
