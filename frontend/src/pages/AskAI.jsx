import axios from "axios";
import { ArrowUpRight, Bot, BookOpen, Leaf, Send, Sparkles } from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";

function normalize(value = "") {
  return value.toLowerCase().replaceAll("_", " ").replaceAll("___", " ").replace(/[^\p{L}\p{N}\s]/gu, " ");
}

export default function AskAI() {
  const [diseases, setDiseases] = useState([]);
  const [crops, setCrops] = useState([]);
  const [question, setQuestion] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    Promise.all([axios.get("/api/diseases"), axios.get("/api/crops")])
      .then(([diseaseResponse, cropResponse]) => {
        setDiseases(diseaseResponse.data);
        setCrops(cropResponse.data);
      })
      .catch(() => setError("Could not load the crop knowledge library. Please try again later."))
      .finally(() => setLoading(false));
  }, []);

  const matches = useMemo(() => {
    if (!submitted) return [];
    const terms = normalize(submitted).split(/\s+/).filter((term) => term.length > 2);
    return diseases
      .map((disease) => {
        const searchable = normalize([
          disease.name,
          disease.crop,
          disease.description,
          ...(disease.symptoms || []),
          ...(disease.treatment || []),
          ...(disease.prevention || []),
        ].join(" "));
        const score = terms.reduce((total, term) => total + (searchable.includes(term) ? 1 : 0), 0);
        return { disease, score };
      })
      .filter((item) => item.score > 0)
      .sort((a, b) => b.score - a.score)
      .slice(0, 3)
      .map((item) => item.disease);
  }, [diseases, submitted]);

  const ask = (event) => {
    event.preventDefault();
    if (!question.trim()) return;
    setSubmitted(question.trim());
    setQuestion("");
  };

  return (
    <div className="page-stack">
      <div className="page-heading-row">
        <div><p className="eyebrow">CROP KNOWLEDGE</p><h1>Ask the crop guide</h1><p className="page-subtitle">Find relevant crop and disease guidance from your library.</p></div>
      </div>
      <section className="assistant-layout">
        <div className="assistant-main panel">
          <div className="assistant-welcome">
            <span className="assistant-icon"><Bot size={23} /></span>
            <div><h2>How can I help with your crops?</h2><p>This guide searches the disease information saved in your library. It does not replace a field inspection or professional advice.</p></div>
          </div>
          {error && <div className="notice notice-error">{error}</div>}
          {loading && <div className="page-loading"><span className="loading-spinner" />Loading crop knowledge…</div>}
          {submitted && !loading && (
            <div className="assistant-response">
              <div className="question-bubble"><span>You asked</span><p>{submitted}</p></div>
              <div className="response-bubble">
                <div className="response-title"><Sparkles size={17} /><strong>Relevant library information</strong></div>
                {matches.length ? (
                  <>
                    <p className="response-intro">These entries contain terms related to your question:</p>
                    <div className="assistant-results">
                      {matches.map((item) => (
                        <article key={item.id} className="assistant-result">
                          <span className="card-overline">{item.crop}</span>
                          <h3>{item.name}</h3>
                          <p>{item.description || "See the full library entry for details."}</p>
                          {item.treatment?.length > 0 && <p><strong>Treatment:</strong> {item.treatment.slice(0, 2).join(" ")}</p>}
                          {item.prevention?.length > 0 && <p><strong>Prevention:</strong> {item.prevention.slice(0, 2).join(" ")}</p>}
                          <Link to="/library" className="text-link">Open disease library <ArrowUpRight size={14} /></Link>
                        </article>
                      ))}
                    </div>
                  </>
                ) : (
                  <p className="response-intro">No matching entries were found for that question. Try a crop name or symptom, or browse the full disease library.</p>
                )}
              </div>
            </div>
          )}
          <form className="assistant-input" onSubmit={ask}>
            <label htmlFor="crop-question" className="sr-only">Ask a question about crop health</label>
            <input id="crop-question" value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about a crop, symptom or disease…" />
            <button type="submit" className="button button-primary" disabled={!question.trim() || loading}><Send size={16} /> Ask</button>
          </form>
        </div>
        <aside className="assistant-side panel">
          <div className="panel-header"><div><p className="card-overline">QUICK START</p><h2>Try asking</h2></div><BookOpen size={18} /></div>
          <div className="suggestion-list">
            {[
              "What are common tomato leaf diseases?",
              "How can I prevent powdery mildew?",
              "What symptoms does apple scab cause?",
            ].map((suggestion) => (
              <button key={suggestion} type="button" onClick={() => { setQuestion(suggestion); setSubmitted(""); }}>
                <Leaf size={15} />{suggestion}<ArrowUpRight size={14} />
              </button>
            ))}
          </div>
          <p className="assistant-count">{loading ? "Loading library…" : `${diseases.length} disease guides · ${crops.length} crops`}</p>
          <Link to="/library" className="text-link">Browse library <ArrowUpRight size={14} /></Link>
        </aside>
      </section>
    </div>
  );
}
