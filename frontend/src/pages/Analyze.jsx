import axios from "axios";
import { Loader2, UploadCloud } from "lucide-react";
import { useRef, useState } from "react";

const severityStyles = {
  None: "bg-emerald-100 text-emerald-800",
  Moderate: "bg-amber-100 text-amber-800",
  High: "bg-orange-100 text-orange-800",
  Critical: "bg-rose-100 text-rose-800",
};

export default function Analyze() {
  const inputRef = useRef(null);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);

  const acceptFile = (selected) => {
    if (!selected) return;
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  const onDrop = (event) => {
    event.preventDefault();
    setDragOver(false);
    acceptFile(event.dataTransfer.files?.[0]);
  };

  const analyze = async () => {
    if (!file) {
      setError("Choose a leaf image first.");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const { data } = await axios.post("/api/predictions/analyze", form, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResult(data);
    } catch (err) {
      setError(err.response?.data?.detail || "Analysis failed. Train the model and start the API.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="mx-auto max-w-6xl px-4 py-8">
      <h1 className="text-3xl font-bold text-leaf-900">Analyze a leaf</h1>
      <p className="mt-2 max-w-2xl text-slate-600">
        Drop a close-up photo of a symptomatic leaf. The backend stores the file, runs MobileNetV2, and returns
        crop, disease, severity, and management steps.
      </p>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <section className="card p-5">
          <div
            onDragOver={(event) => {
              event.preventDefault();
              setDragOver(true);
            }}
            onDragLeave={() => setDragOver(false)}
            onDrop={onDrop}
            className={`grid min-h-[260px] place-items-center rounded-2xl border-2 border-dashed p-6 text-center ${
              dragOver ? "border-leaf-600 bg-leaf-50" : "border-slate-300 bg-slate-50"
            }`}
          >
            {preview ? (
              <img src={preview} alt="Leaf preview" className="max-h-64 rounded-xl object-contain" />
            ) : (
              <div>
                <UploadCloud className="mx-auto text-leaf-600" />
                <p className="mt-2 font-medium">Drag and drop a leaf image</p>
                <p className="text-sm text-slate-500">JPEG, PNG, or WebP up to 8 MB</p>
              </div>
            )}
          </div>
          <div className="mt-4 flex flex-wrap gap-3">
            <button
              type="button"
              onClick={() => inputRef.current?.click()}
              className="rounded-xl border border-slate-200 px-4 py-2 text-sm font-medium"
            >
              Choose file
            </button>
            <button
              type="button"
              onClick={analyze}
              disabled={loading}
              className="inline-flex items-center gap-2 rounded-xl bg-leaf-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-60"
            >
              {loading && <Loader2 className="animate-spin" size={16} />}
              {loading ? "Running diagnosis..." : "Analyze leaf"}
            </button>
            <input
              ref={inputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={(event) => acceptFile(event.target.files?.[0])}
            />
          </div>
          {error && <p className="mt-3 text-sm text-rose-700">{error}</p>}
        </section>

        <section className="card p-5">
          {loading && (
            <div className="flex h-full min-h-[260px] flex-col items-center justify-center gap-3 text-leaf-700">
              <Loader2 className="h-10 w-10 animate-spin" />
              <p className="font-medium">Extracting leaf features with MobileNetV2</p>
            </div>
          )}
          {!loading && !result && (
            <div className="grid min-h-[260px] place-items-center text-slate-500">
              Diagnostic details will appear here after analysis.
            </div>
          )}
          {!loading && result && (
            <div className="space-y-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm text-slate-500">Crop</p>
                  <h2 className="text-2xl font-bold">{result.crop}</h2>
                  <p className="text-lg text-leaf-700">{result.disease}</p>
                </div>
                <span className={`rounded-full px-3 py-1 text-xs font-semibold ${severityStyles[result.severity] || "bg-slate-100"}`}>
                  {result.severity} severity
                </span>
              </div>
              <div>
                <div className="mb-1 flex justify-between text-sm">
                  <span>Confidence</span>
                  <span>{result.confidence}%</span>
                </div>
                <div className="h-2 overflow-hidden rounded-full bg-slate-100">
                  <div className="h-full rounded-full bg-leaf-600" style={{ width: `${result.confidence}%` }} />
                </div>
              </div>
              <div>
                <h3 className="font-semibold">Top-3 classes</h3>
                <ul className="mt-1 space-y-1 text-sm text-slate-600">
                  {result.top3.map((item) => (
                    <li key={item.label} className="flex justify-between">
                      <span>{item.label.replaceAll("___", " / ").replaceAll("_", " ")}</span>
                      <span>{item.confidence}%</span>
                    </li>
                  ))}
                </ul>
              </div>
              {[
                ["Symptoms", result.symptoms],
                ["Treatment", result.treatment],
                ["Prevention", result.prevention],
              ].map(([title, items]) => (
                <div key={title}>
                  <h3 className="font-semibold">{title}</h3>
                  <ul className="mt-1 list-disc space-y-1 pl-5 text-sm text-slate-600">
                    {items.map((item) => (
                      <li key={item}>{item}</li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
