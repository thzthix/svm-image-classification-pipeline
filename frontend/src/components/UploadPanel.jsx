function UploadPanel({
  selectedFile,
  previewUrl,
  topK,
  isPredictLoading,
  isSimilarLoading,
  errorMessage,
  onFileChange,
  onTopKChange,
  onPredict,
  onSimilar,
}) {
  const isPredictDisabled = !selectedFile || isPredictLoading;
  const isSimilarDisabled = !selectedFile || isSimilarLoading;

  return (
    <section className="panel upload-panel">
      <div className="panel-header">
        <h2>이미지 업로드</h2>
        <p>의류 이미지를 선택한 뒤 원하는 기능을 실행해 주세요.</p>
      </div>

      <label className="field">
        <span>이미지 파일</span>
        <input
          type="file"
          accept="image/png,image/jpeg,image/jpg"
          onChange={onFileChange}
        />
      </label>

      <div className="preview-box">
        {previewUrl ? (
          <img src={previewUrl} alt="업로드 미리보기" className="preview-image" />
        ) : (
          <div className="empty-preview">업로드한 이미지가 여기에 표시됩니다.</div>
        )}
      </div>

      <label className="field">
        <span>비슷한 이미지 개수</span>
        <input type="number" min="1" max="20" value={topK} onChange={onTopKChange} />
      </label>

      <div className="button-row">
        <button type="button" onClick={onPredict} disabled={isPredictDisabled}>
          {isPredictLoading ? "분류 중..." : "분류하기"}
        </button>
        <button
          type="button"
          className="secondary-button"
          onClick={onSimilar}
          disabled={isSimilarDisabled}
        >
          {isSimilarLoading ? "검색 중..." : "비슷한 이미지 찾기"}
        </button>
      </div>

      {errorMessage ? <p className="status-text is-error">{errorMessage}</p> : null}
    </section>
  );
}

export default UploadPanel;
