import { useState } from "react";
import QueryForm from "./components/QueryForm";
import CourseList from "./components/CourseList";
import { queryCourses } from "./api";
import "./index.css";

const App = () => {
  const [answer, setAnswer] = useState("");
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(false);

  const handleQuery = async (query, department) => {
    setLoading(true);
    setAnswer("");
    setCourses([]);

    try {
      const result = await queryCourses(query, department);
      setAnswer(result.answer || "No answer generated.");
      setCourses(result.courses || []);
    } catch (error) {
      setAnswer("Error fetching results. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>AI-Powered Brown Course Search</h1>
      <QueryForm onSubmit={handleQuery} loading={loading} />
      {answer && (
        <div className="answer-box">
          <h3>AI Generated Answer</h3>
          <p>{answer}</p>
        </div>
      )}
      <CourseList courses={courses} />
    </div>
  );
};

export default App;
