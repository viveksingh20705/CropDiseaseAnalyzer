import axios from "axios";
import { Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

export default function History() {
  const [items, setItems] = useState([]);
  const [error, setError] = useState("");

  const load = async () => {
    try {
      const { data } = await axios.get("/api/predictions/history");
      setItems(data);
    } catch {
      setError("Could not load analysis history.");
    }
  };

  useEffect(() => {
    load();
  }, []);

  const remove = async (id) => {
    await axios.delete(`/api/predictions/${id}`);
    setItems((current) => current.filter((item) => item.id !== id));
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-3xl font-bold text-leaf-900">History</h1>
      <p className="mt-2 text-slate-600">Every uploaded leaf and its stored diagnosis.</p>
      {error && <p className="mt-4 text-rose-700">{error}</p>}
      <div className="mt-6 space-y-4">
        {items.length === 0 && <p className="text-slate-500">No analyses yet. Run one from the Analyze page.</p>}
        {items.map((item) => (
          <article key={item.id} className="card flex flex-col gap-4 p-4 md:flex-row">
            <img
              src={item.image_url}
              alt={`${item.crop} ${item.disease}`}
              className="h-36 w-full rounded-xl object-cover md:w-48"
            />
            <div className="flex-1">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-lg font-semibold">
                    {item.crop} — {item.disease}
                  </h2>
                  <p className="text-sm text-slate-500">
                    {new Date(item.created_at).toLocaleString()} · {item.confidence}% confidence
                  </p>
                </div>
                <button type="button" onClick={() => remove(item.id)} className="rounded-lg p-2 text-rose-600 hover:bg-rose-50">
                  <Trash2 size={18} />
                </button>
              </div>
              <p className="mt-2 text-sm text-slate-600">{item.symptoms[0]}</p>
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}
