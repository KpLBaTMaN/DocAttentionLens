import { useEffect, useState } from 'react';
import { errorMessage } from '../api/client';

export function useRemote<T>(load: () => Promise<T>, interval?: number) {
  const [data, setData] = useState<T>();
  const [error, setError] = useState<string>();
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const result = await load();
        if (alive) { setData(result); setError(undefined); }
      } catch (reason) { if (alive) setError(errorMessage(reason)); }
      if (alive && interval) timer = setTimeout(refresh, interval);
    }
    setData(undefined);
    setError(undefined);
    void refresh();
    return () => { alive = false; clearTimeout(timer); };
  }, [load, interval]);
  return { data, error };
}
