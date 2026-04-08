// This is a placeholder for actual backend connections to FastAPI
export const MOCK_API = {
  uploadDocument: async (file) => {
    // Expected to connect to POST /upload
    return new Promise(resolve => setTimeout(() => resolve({ success: true }), 1500));
  },
  queryTutor: async (query, contextFiles) => {
    // Expected to connect to POST /query
    return new Promise(resolve => setTimeout(() => resolve({ 
      answer: "Mock grounded answer",
      citations: [contextFiles[0]?.name || "Reference.pdf"]
    }), 2000));
  },
  generateAssessment: async (type, contextFiles) => {
    // Expected to connect to POST /generate_assessment
    return new Promise(resolve => setTimeout(() => resolve({ items: [] }), 2500));
  }
};
