function PredictionResult({ predictResult }) {
  return (
    <section className="panel prediction-panel">
      <div className="panel-header">
        <h2>분류 결과</h2>
        <p>예측된 카테고리와 결정 점수를 확인할 수 있습니다.</p>
      </div>

      <div className="metric-grid metric-grid-compact">
        <article className="metric-card">
          <span className="metric-label">예측 레이블</span>
          <strong className="metric-value">
            {predictResult?.predicted_label ?? "-"}
          </strong>
        </article>

        <article className="metric-card">
          <span className="metric-label">결정 점수</span>
          <strong className="metric-value">
            {formatDecisionScore(predictResult?.decision_score)}
          </strong>
        </article>
      </div>
    </section>
  );
}

function formatDecisionScore(score) {
  if (typeof score !== "number") {
    return "-";
  }

  return score.toFixed(4);
}

export default PredictionResult;
