import { useEffect, useState } from "react";
import PredictionResult from "./components/PredictionResult";
import SimilarResults from "./components/SimilarResults";
import UploadPanel from "./components/UploadPanel";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL ?? "").replace(/\/$/, "");

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [topK, setTopK] = useState(5);
  const [predictResult, setPredictResult] = useState(null);
  const [similarResults, setSimilarResults] = useState([]);
  const [isPredictLoading, setIsPredictLoading] = useState(false);
  const [isSimilarLoading, setIsSimilarLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    if (!selectedFile) {
      setPreviewUrl("");
      return;
    }

    const nextPreviewUrl = URL.createObjectURL(selectedFile);
    setPreviewUrl(nextPreviewUrl);

    return () => {
      URL.revokeObjectURL(nextPreviewUrl);
    };
  }, [selectedFile]);

  function handleFileChange(event) {
    const nextFile = event.target.files?.[0] ?? null;
    setSelectedFile(nextFile);
    setPredictResult(null);
    setSimilarResults([]);
    setErrorMessage("");
  }

  function handleTopKChange(event) {
    const nextValue = Number(event.target.value);
    if (Number.isNaN(nextValue)) {
      setTopK(1);
      return;
    }

    setTopK(Math.min(Math.max(nextValue, 1), 20));
  }

  function createFormData(includeTopK = false) {
    const formData = new FormData();

    if (selectedFile) {
      formData.append("file", selectedFile);
    }

    if (includeTopK) {
      formData.append("top_k", String(topK));
    }

    return formData;
  }

  async function requestJson(path, formData) {
    const response = await fetch(`${API_BASE_URL}${path}`, {
      method: "POST",
      body: formData,
    });

    const payload = await response.json().catch(() => null);

    if (!response.ok) {
      throw new Error(payload?.detail ?? "요청 처리 중 오류가 발생했습니다.");
    }

    return payload;
  }

  async function handlePredict() {
    if (!selectedFile) {
      setErrorMessage("먼저 이미지를 업로드해 주세요.");
      return;
    }

    setIsPredictLoading(true);
    setErrorMessage("");

    try {
      const result = await requestJson("/predict", createFormData());
      setPredictResult(result);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsPredictLoading(false);
    }
  }

  async function handleSimilar() {
    if (!selectedFile) {
      setErrorMessage("먼저 이미지를 업로드해 주세요.");
      return;
    }

    setIsSimilarLoading(true);
    setErrorMessage("");

    try {
      const result = await requestJson("/similar", createFormData(true));
      setSimilarResults(result);
    } catch (error) {
      setErrorMessage(error.message);
    } finally {
      setIsSimilarLoading(false);
    }
  }

  return (
    <main className="page">
      <section className="hero">
        <h1>SVM Image Classification Demo</h1>
        <p className="hero-copy">
          의류 이미지를 업로드하면 모델이 해당 이미지가 어떤 카테고리에 속하는지
          분류합니다. 또한 비슷한 이미지를 검색할 수 있습니다.
        </p>
      </section>

      <section className="layout">
        <UploadPanel
          selectedFile={selectedFile}
          previewUrl={previewUrl}
          topK={topK}
          isPredictLoading={isPredictLoading}
          isSimilarLoading={isSimilarLoading}
          errorMessage={errorMessage}
          onFileChange={handleFileChange}
          onTopKChange={handleTopKChange}
          onPredict={handlePredict}
          onSimilar={handleSimilar}
        />
        <section className="result-column">
          <PredictionResult predictResult={predictResult} />
          <SimilarResults similarResults={similarResults} />
        </section>
      </section>
    </main>
  );
}

export default App;
