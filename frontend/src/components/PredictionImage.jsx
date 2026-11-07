import { Leaf } from "lucide-react";
import { useState } from "react";

export default function PredictionImage({ src, alt, className = "" }) {
  const [unavailable, setUnavailable] = useState(!src);

  if (unavailable) {
    return (
      <span className={`prediction-image-fallback ${className}`} role="img" aria-label={alt}>
        <Leaf size={19} />
      </span>
    );
  }

  return (
    <img
      src={src}
      alt={alt}
      className={className}
      onError={() => setUnavailable(true)}
    />
  );
}
