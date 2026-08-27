import axios from "axios";
import { useEffect, useState } from "react";

export default function Library() {
  const [diseases, setDiseases] = useState([]);
  const [crops, setCrops] = useState([]);

  useEffect(() => {
    axios.get("/api/diseases").then(({ data }) => setDiseases(data));
    axios.get("/api/crops").then(({ data }) => setCrops(data));
  }, []);

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-3xl font-bold text-leaf-900">Disease library</h1>
      <p className="mt-2 text-slate-600">Reference crops and management notes seeded into MongoDB.</p>
      <div className="mt-6 flex flex-wrap gap-2">
        {crops.map((crop) => (
          <span key={crop.id} className="rounded-full bg-leaf-50 px-3 py-1 text-sm text-leaf-800">
            {crop.name}
          </span>
        ))}
      </div>
      <div className="mt-6 grid gap-4 md:grid-cols-2">
        {diseases.map((disease) => (
          <article key={disease.id} className="card p-5">
            <p className="text-xs uppercase tracking-wide text-leaf-600">{disease.crop}</p>
            <h2 className="text-xl font-semibold">{disease.name}</h2>
            <p className="mt-1 text-sm text-slate-500">{disease.severity} severity</p>
            <p className="mt-2 text-sm text-slate-600">{disease.description}</p>
            <h3 className="mt-3 text-sm font-semibold">Symptoms</h3>
            <ul className="list-disc pl-5 text-sm text-slate-600">
              {disease.symptoms.map((item) => (
                <li key={item}>{item}</li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </div>
  );
}
