import axios from "axios";

const API_BASE_URL = "http://localhost:8000";

export const queryCourses = async (q, department = "") => {
  try {
    const response = await axios.post(`${API_BASE_URL}/query`, { q, department });
    return response.data;
  } catch (err) {
    console.error("Error fetching query:", err);
    throw err;
  }
};