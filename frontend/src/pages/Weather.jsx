import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSun,
  Droplets,
  MapPin,
  RefreshCw,
  Sun,
  Wind,
} from "lucide-react";
import { useState } from "react";

function weatherDetails(code) {
  if (code === 0) return ["Clear sky", Sun];
  if ([1, 2].includes(code)) return ["Partly cloudy", CloudSun];
  if (code === 3) return ["Overcast", Cloud];
  if ([45, 48].includes(code)) return ["Foggy", CloudFog];
  if ([51, 53, 55, 56, 57].includes(code)) return ["Drizzle", CloudDrizzle];
  if ([61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return ["Rain", CloudRain];
  if ([71, 73, 75, 77, 85, 86].includes(code)) return ["Snow", Cloud];
  if ([95, 96, 99].includes(code)) return ["Thunderstorm", CloudLightning];
  return ["Current conditions", CloudSun];
}

export default function Weather() {
  const [weather, setWeather] = useState(null);
  const [place, setPlace] = useState("");
  const [status, setStatus] = useState("");
  const [loading, setLoading] = useState(false);

  const loadWeather = () => {
    if (!navigator.geolocation) {
      setStatus("Location access is not available in this browser.");
      return;
    }
    setLoading(true);
    setStatus("");
    navigator.geolocation.getCurrentPosition(async ({ coords }) => {
      try {
        const params = new URLSearchParams({
          latitude: String(coords.latitude),
          longitude: String(coords.longitude),
          current: "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
          timezone: "auto",
        });
        const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
        if (!response.ok) throw new Error("Weather service is temporarily unavailable.");
        const data = await response.json();
        setWeather(data.current);
        setPlace(data.timezone?.split("/")?.pop()?.replaceAll("_", " ") || "Your location");
      } catch (error) {
        setStatus(error.message || "Could not load local weather.");
      } finally {
        setLoading(false);
      }
    }, (error) => {
      setStatus(error.code === 1 ? "Allow location access in your browser to see local weather." : "Could not determine your location.");
      setLoading(false);
    }, { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 });
  };

  const [condition, WeatherIcon] = weather ? weatherDetails(weather.weather_code) : ["", CloudSun];

  return (
    <div className="page-stack">
      <div className="page-heading-row">
        <div><p className="eyebrow">LOCAL CONDITIONS</p><h1>Weather</h1><p className="page-subtitle">Check the current conditions near your farm.</p></div>
        <button type="button" className="button button-secondary" onClick={loadWeather} disabled={loading}>
          <RefreshCw size={16} className={loading ? "spin" : ""} /> {weather ? "Refresh" : "Use my location"}
        </button>
      </div>
      <section className="weather-card">
        {weather ? (
          <>
            <div className="weather-main">
              <div><span className="weather-place"><MapPin size={15} />{place}</span><h2>{condition}</h2><p>Current local weather</p></div>
              <WeatherIcon className="weather-main-icon" size={72} strokeWidth={1.2} />
              <strong className="weather-temp">{Math.round(weather.temperature_2m)}°<small>C</small></strong>
            </div>
            <div className="weather-details">
              <div><Droplets size={18} /><span>Humidity</span><strong>{weather.relative_humidity_2m}%</strong></div>
              <div><Wind size={18} /><span>Wind speed</span><strong>{weather.wind_speed_10m} km/h</strong></div>
              <div><CloudRain size={18} /><span>Precipitation</span><strong>{weather.precipitation} mm</strong></div>
            </div>
            <p className="weather-source">Live conditions from Open-Meteo · Updated {weather.time?.replace("T", " ")}</p>
          </>
        ) : (
          <div className="weather-empty">
            <span className="weather-empty-icon"><CloudSun size={31} /></span>
            <h2>Weather for your location</h2>
            <p>Share your location to view current temperature, humidity, wind and precipitation. Your coordinates are used only to request weather and aren’t saved.</p>
            {status && <div className="notice notice-error">{status}</div>}
            <button type="button" className="button button-primary" onClick={loadWeather} disabled={loading}>
              <MapPin size={16} /> {loading ? "Finding your location…" : "Get local weather"}
            </button>
          </div>
        )}
      </section>
      <div className="notice notice-info"><CloudSun size={18} /><span>Weather can change quickly. Check local forecasts before making irrigation or treatment decisions.</span></div>
    </div>
  );
}
