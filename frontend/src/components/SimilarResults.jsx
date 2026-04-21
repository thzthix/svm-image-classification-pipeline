function SimilarResults({ similarResults }) {
  return (
    <section className="panel similar-panel">
      <div className="panel-header">
        <h2>비슷한 이미지</h2>
        <p>업로드한 이미지와 가까운 결과를 확인할 수 있습니다.</p>
      </div>

      <div className="similar-results-box">
        {similarResults.length === 0 ? (
          <p className="empty-text">아직 결과가 없습니다.</p>
        ) : (
          <ul className="similar-list">
            {similarResults.map((item) => (
              <li key={`${item.index}-${item.label}`} className="similar-item">
                <div className="similar-thumbnail-box">
                  {item.thumbnail_data_url ? (
                    <img
                      src={item.thumbnail_data_url}
                      alt={`${item.label_name} 썸네일`}
                      className="similar-thumbnail"
                    />
                  ) : (
                    <div className="similar-thumbnail-placeholder" />
                  )}
                </div>
                <div className="similar-content">
                  <strong className="similar-label">{item.label_name}</strong>
                  <span className="similar-score">
                    유사도 {formatSimilarityScore(item.score)}
                  </span>
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>
    </section>
  );
}

function formatSimilarityScore(score) {
  if (typeof score !== "number") {
    return "-";
  }

  return `${(score * 100).toFixed(2)}%`;
}

export default SimilarResults;
