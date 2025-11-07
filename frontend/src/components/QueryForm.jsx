import { useState }  from "react";

const QueryForm = ({ onSubmit, loading }) => {
  const [query, setQuery] = useState("");
  const [department, setDepartment] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!query.trim()) return;
    onSubmit(query, department);
  };

  return (
    <form className="query-form" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="Ask about courses (e.g. 'Who teaches ECON 2930?')"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        disabled={loading}
      />
      <input
        type="text"
        placeholder="Optional: Department (e.g. CSCI)"
        value={department}
        onChange={(e) => setDepartment(e.target.value)}
        disabled={loading}
      />
      <button type="submit" disabled={loading}>
        {loading ? "Searching..." : "Search"}
      </button>
    </form>
  );
};

export default QueryForm;
