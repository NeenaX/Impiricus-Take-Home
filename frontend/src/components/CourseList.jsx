const CourseList = ({ courses }) => {
  if (!courses?.length) return null;

  return (
    <div className="course-list">
      <h3>Top Results</h3>
      {courses.map((c, idx) => (
        <div key={idx} className="course-card">
          <div className="course-header">
            <strong>{c.title}</strong> <span>({c.course_code})</span>
          </div>
          <p><b>Department:</b> {c.department}</p>
          <p><b>Instructor:</b> {c.instructor}</p>
          <p><b>Source:</b> {c.source}</p>
          <p><b>Similarity:</b> {c.similarity.toFixed(3)}</p>
          <p className="desc">{c.description}</p>
        </div>
      ))}
    </div>
  );
};

export default CourseList;
