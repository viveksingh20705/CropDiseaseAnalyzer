import {
  Cloud,
  CloudDrizzle,
  CloudFog,
  CloudLightning,
  CloudRain,
  CloudSun,
  Droplets,
  LocateFixed,
  MapPin,
  RefreshCw,
  Sun,
  Wind,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

const MAX_LOCATION_ACCURACY_METERS = 20_000;
const GORAKHPUR_LOCATION = {
  latitude: 26.7606,
  longitude: 83.3732,
  label: "Gorakhpur, Uttar Pradesh, India (fallback)",
};

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
  const locationRequested = useRef(false);

  const fetchWeather = useCallback(async (latitude, longitude, placeName = "") => {
    setLoading(true);
    setStatus("");
    setWeather(null);
    setPlace("Finding your city…");
    try {
      const params = new URLSearchParams({
        latitude: String(latitude),
        longitude: String(longitude),
        current: "temperature_2m,relative_humidity_2m,apparent_temperature,is_day,precipitation,weather_code,wind_speed_10m",
        timezone: "auto",
      });
      const response = await fetch(`https://api.open-meteo.com/v1/forecast?${params}`);
      if (!response.ok) throw new Error("Weather service is temporarily unavailable.");
      const data = await response.json();
      setWeather(data.current);
    } catch (error) {
      setStatus(error.message || "Could not load weather for your location.");
      setLoading(false);
      return;
    }

    if (placeName) {
      setPlace(placeName);
      setLoading(false);
      return;
    }

    try {
      const locationParams = new URLSearchParams({
        latitude: String(latitude),
        longitude: String(longitude),
        localityLanguage: "en",
      });
      const locationResponse = await fetch(
        `https://api.bigdatacloud.net/data/reverse-geocode-client?${locationParams}`
      );
      if (!locationResponse.ok) throw new Error("Location lookup failed.");
      const location = await locationResponse.json();
      const city = location.city || location.locality;
      const name = [city, location.principalSubdivision, location.countryName]
        .filter((value, index, values) => value && values.indexOf(value) === index)
        .join(", ");
      if (name) setPlace(name);
      else {
        setPlace("Your current location");
        setStatus("Weather loaded, but your city name could not be identified.");
      }
    } catch {
      setPlace("Your current location");
      setStatus("Weather loaded, but your city name could not be identified.");
    } finally {
      setLoading(false);
    }
  }, []);

  const fetchFallbackWeather = useCallback(
    () => fetchWeather(GORAKHPUR_LOCATION.latitude, GORAKHPUR_LOCATION.longitude, GORAKHPUR_LOCATION.label),
    [fetchWeather]
  );

  const loadWeather = useCallback(() => {
    if (!navigator.geolocation) {
      fetchFallbackWeather();
      return;
    }
    setLoading(true);
    setStatus("");
    navigator.geolocation.getCurrentPosition(({ coords }) => {
      if (coords.accuracy > MAX_LOCATION_ACCURACY_METERS) {
        fetchFallbackWeather();
      } else {
        fetchWeather(coords.latitude, coords.longitude);
      }
    }, () => {
      fetchFallbackWeather();
    }, { enableHighAccuracy: true, timeout: 20000, maximumAge: 0 });
  }, [fetchFallbackWeather, fetchWeather]);

  useEffect(() => {
    if (locationRequested.current) return;
    locationRequested.current = true;
    loadWeather();
  }, [loadWeather]);

  const [condition, WeatherIcon] = weather ? weatherDetails(weather.weather_code) : ["", CloudSun];

  return (
    <div className="page-stack">
      <div className="page-heading-row">
        <div><p className="eyebrow">LOCAL CONDITIONS</p><h1>Weather</h1><p className="page-subtitle">Uses precise device location, or Gorakhpur if location is unavailable.</p></div>
        <button type="button" className="button button-secondary" onClick={loadWeather} disabled={loading}>
          <RefreshCw size={16} className={loading ? "spin" : ""} /> Refresh
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
            <span className="weather-empty-icon">{loading ? <LocateFixed size={31} className="spin" /> : <CloudSun size={31} />}</span>
            <h2>{loading ? "Finding your location…" : "Weather for your location"}</h2>
            <p>{loading
              ? "Getting your device location and current local conditions."
              : "Allow location access to see current temperature, humidity, wind and precipitation. Your coordinates aren’t saved."}</p>
            {status && <div className="notice notice-error">{status}</div>}
            {!loading && <button type="button" className="button button-primary" onClick={loadWeather}><LocateFixed size={16} /> Try location again</button>}
          </div>
        )}
      </section>
      {status && weather && <div className="notice notice-error" role="alert">{status}</div>}
      <div className="notice notice-info"><CloudSun size={18} /><span>Weather can change quickly. Check local forecasts before making irrigation or treatment decisions.</span></div>
    </div>
  );
}
