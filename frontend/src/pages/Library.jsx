import axios from "axios";
import { useEffect, useState } from "react";

export default function Library() {
  const [diseases, setDiseases] = useState([]);
  const [crops, setCrops] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([axios.get("/api/diseases"), axios.get("/api/crops")])
      .then(([diseaseResponse, cropResponse]) => {
        setDiseases(diseaseResponse.data);
        setCrops(cropResponse.data);
      })
      .catch(() => setError("Could not load the disease library. Please try again later."))
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="page-stack legacy-page">
      <h1 className="text-3xl font-bold text-leaf-900">Disease library</h1>
      <p className="mt-2 text-slate-600">Reference crops and management notes seeded into MongoDB.</p>
      <div className="mt-6 flex flex-wrap gap-2">
        {crops.map((crop) => (
          <span key={crop.id} className="rounded-full bg-leaf-50 px-3 py-1 text-sm text-leaf-800">
            {crop.name}
          </span>
        ))}
      </div>
      {error && <p role="alert" className="text-sm text-rose-700">{error}</p>}
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {loading && <p className="text-sm text-slate-500">Loading disease information…</p>}
        {!loading && !error && diseases.length === 0 && (
          <p className="text-sm text-slate-500">No disease entries are available yet.</p>
        )}
        {diseases.map((disease) => (
          <article key={disease.id} className="card p-5">
            <p className="text-xs uppercase tracking-wide text-leaf-600">{disease.crop}</p>
            <h2 className="text-xl font-semibold">{disease.name}</h2>
            <p className="mt-1 text-sm text-slate-500">{disease.severity} severity</p>
            {disease.description && <p className="mt-2 text-sm text-slate-600">{disease.description}</p>}
            {[["Symptoms", disease.symptoms], ["Treatment", disease.treatment], ["Prevention", disease.prevention]].map(([title, items]) => (
              items?.length > 0 && (
                <div key={title}>
                  <h3 className="mt-3 text-sm font-semibold">{title}</h3>
                  <ul className="list-disc pl-5 text-sm text-slate-600">
                    {items.map((item) => <li key={item}>{item}</li>)}
                  </ul>
                </div>
              )
            ))}
          </article>
        ))}
      </div>
    </div>
  );
}
