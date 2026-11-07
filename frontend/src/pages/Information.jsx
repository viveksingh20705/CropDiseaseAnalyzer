import { ArrowUpRight, BookOpen, Mail, MessageCircle } from "lucide-react";
import { Link } from "react-router-dom";

export function Help() {
  return (
    <div className="page-stack">
      <div className="page-heading-row"><div><p className="eyebrow">WE’RE HERE TO HELP</p><h1>Help &amp; Support</h1><p className="page-subtitle">Get the most from your crop disease analyzer.</p></div></div>
      <section className="info-card-grid">
        <article className="panel info-card"><span className="settings-icon"><BookOpen size={19} /></span><h2>Using the disease library</h2><p>Browse documented crop diseases, symptoms, treatment notes and prevention guidance from your app’s reference library.</p><Link className="text-link" to="/library">Explore the library <ArrowUpRight size={14} /></Link></article>
        <article className="panel info-card"><span className="settings-icon settings-icon-blue"><MessageCircle size={19} /></span><h2>Need diagnosis help?</h2><p>Upload a clear, well-lit leaf photo. AI predictions are a helpful starting point and should be confirmed with local expertise.</p><Link className="text-link" to="/analyze">Analyze a crop <ArrowUpRight size={14} /></Link></article>
        <article className="panel info-card"><span className="settings-icon settings-icon-amber"><Mail size={19} /></span><h2>Account support</h2><p>Review your account information and security preferences from Settings.</p><Link className="text-link" to="/settings">Open settings <ArrowUpRight size={14} /></Link></article>
      </section>
    </div>
  );
}

export function About() {
  return (
    <div className="page-stack">
      <div className="page-heading-row"><div><p className="eyebrow">ABOUT THIS TOOL</p><h1>Crop Disease Analyzer</h1><p className="page-subtitle">AI-assisted crop leaf analysis and disease information.</p></div></div>
      <section className="panel about-card"><span className="about-logo"><BookOpen size={23} /></span><h2>Supporting healthier crops with accessible diagnostics</h2><p>Crop Disease Analyzer uses a MobileNetV2 image classification model to analyze uploaded leaf images. Results include a predicted crop and condition, confidence, and related management information stored in your account’s prediction history.</p><p>Predictions are informational and may be affected by image quality, crop variety, and environmental conditions. Confirm important treatment decisions with a qualified local agricultural specialist.</p><div className="about-links"><Link to="/analyze" className="button button-primary">Analyze an image</Link><Link to="/library" className="button button-secondary">Browse disease library</Link></div></section>
    </div>
  );
}
