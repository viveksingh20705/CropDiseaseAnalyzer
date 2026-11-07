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
  Search,
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
  const [searching, setSearching] = useState(false);
  const [locationQuery, setLocationQuery] = useState("");
  const [locations, setLocations] = useState([]);
  const [locationError, setLocationError] = useState("");

  const fetchWeather = async (latitude, longitude, locationName) => {
    setLoading(true);
    setStatus("");
    setWeather(null);
    setPlace(locationName);
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
      setLocations([]);
      setLocationError("");
    } catch (error) {
      setStatus(error.message || "Could not load weather for this location.");
    } finally {
      setLoading(false);
    }
  };

  const searchLocations = async (event) => {
    event.preventDefault();
    const query = locationQuery.trim();
    if (!query) {
      setLocationError("Enter a city or town to search.");
      setLocations([]);
      return;
    }
    setSearching(true);
    setLocationError("");
    setLocations([]);
    try {
      const params = new URLSearchParams({
        name: query,
        count: "8",
        language: "en",
        format: "json",
      });
      const response = await fetch(`https://geocoding-api.open-meteo.com/v1/search?${params}`);
      if (!response.ok) throw new Error("Location search is temporarily unavailable.");
      const data = await response.json();
      const matches = data.results || [];
      if (!matches.length) {
        setLocationError("No matching city or town found. Try another spelling.");
        return;
      }
      if (matches.length === 1) {
        const match = matches[0];
        await fetchWeather(match.latitude, match.longitude, [match.name, match.admin1, match.country].filter(Boolean).join(", "));
        return;
      }
      setLocations(matches);
    } catch (error) {
      setLocationError(error.message || "Could not search for that location.");
    } finally {
      setSearching(false);
    }
  };

  const chooseLocation = (location) => {
    fetchWeather(
      location.latitude,
      location.longitude,
      [location.name, location.admin1, location.country].filter(Boolean).join(", ")
    );
  };

  const loadWeather = () => {
    if (!navigator.geolocation) {
      setStatus("Location access is unavailable. Search for your city instead.");
      return;
    }
    setLoading(true);
    setStatus("");
    navigator.geolocation.getCurrentPosition(({ coords }) => {
      fetchWeather(coords.latitude, coords.longitude, "Near your current location");
    }, (error) => {
      setStatus(error.code === 1
        ? "Location access was denied. Search for Gorakhpur or another city instead."
        : "Could not determine your location. Search for your city instead.");
      setLoading(false);
    }, { enableHighAccuracy: false, timeout: 10000, maximumAge: 300000 });
  };

  const [condition, WeatherIcon] = weather ? weatherDetails(weather.weather_code) : ["", CloudSun];

  return (
    <div className="page-stack">
      <div className="page-heading-row">
        <div><p className="eyebrow">LOCAL CONDITIONS</p><h1>Weather</h1><p className="page-subtitle">Search for your city or use your device location.</p></div>
        <button type="button" className="button button-secondary" onClick={loadWeather} disabled={loading}>
          <LocateFixed size={16} className={loading ? "spin" : ""} /> Use my location
        </button>
      </div>
      <section className="weather-search-panel">
        <form className="weather-search-form" onSubmit={searchLocations}>
          <label htmlFor="weather-location"><MapPin size={16} /> Search for your city</label>
          <div className="weather-search-controls">
            <input
              id="weather-location"
              type="search"
              autoComplete="address-level2"
              placeholder="For example, Gorakhpur"
              value={locationQuery}
              onChange={(event) => setLocationQuery(event.target.value)}
            />
            <button type="submit" className="button button-primary" disabled={searching || loading}>
              <Search size={16} /> {searching ? "Searching…" : "Search"}
            </button>
          </div>
        </form>
        {locationError && <p className="weather-search-message" role="alert">{locationError}</p>}
        {locations.length > 0 && (
          <div className="weather-location-results" aria-label="Choose a matching location">
            {locations.map((location) => (
              <button
                type="button"
                key={`${location.id}-${location.latitude}-${location.longitude}`}
                onClick={() => chooseLocation(location)}
                disabled={loading}
              >
                <MapPin size={16} />
                <span><strong>{location.name}</strong><small>{[location.admin1, location.country].filter(Boolean).join(", ")}</small></span>
                <span className="location-result-select">View weather</span>
              </button>
            ))}
          </div>
        )}
      </section>
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
            <p>Search for your city above, or share your device location to view current temperature, humidity, wind and precipitation. Coordinates aren’t saved.</p>
            {status && <div className="notice notice-error">{status}</div>}
            <button type="button" className="button button-primary" onClick={loadWeather} disabled={loading}>
              <MapPin size={16} /> {loading ? "Finding your location…" : "Get local weather"}
            </button>
          </div>
        )}
      </section>
      {status && weather && <div className="notice notice-error" role="alert">{status}</div>}
      <div className="notice notice-info"><CloudSun size={18} /><span>Weather can change quickly. Check local forecasts before making irrigation or treatment decisions.</span></div>
    </div>
  );
}
