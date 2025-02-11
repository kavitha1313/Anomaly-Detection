// import React, { useEffect } from 'react';
// import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
// import Navbar from './components/Navbar';
// import HomePage from './components/pages/HomePage';
// import VideoUploadPage from './components/pages/VideoUploadPage';

// // Scroll to top on route change
// function ScrollToTop() {
//   const { pathname } = useLocation();

//   useEffect(() => {
//     window.scrollTo(0, 0);
//   }, [pathname]);

//   return null;
// }

// // Default export App component
// export default function App() {
//   return (
//     <Router>
//       <ScrollToTop />
//       <div style={styles.appContainer}>
//         <Navbar />
//         <div style={styles.contentContainer}>
//           <Routes>
//             <Route path="/" element={<HomePage />} />
//             <Route path="/upload" element={<VideoUploadPage />} />
//           </Routes>
//         </div>
//       </div>
//     </Router>
//   );
// }

// // Inline styles for App.js
// const styles = {
//   appContainer: {
//     minHeight: '100vh',
//     backgroundColor: '#f9fafb', // Equivalent to Tailwind's bg-gray-50
//   },
//   contentContainer: {
//     paddingTop: '64px', // Adjust for fixed navbar height
//   },
// };






// import React, { useEffect } from 'react';
// import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
// import Navbar from './components/Navbar';
//  import HomePage from './components/pages/HomePage';
// import VideoUploadPage from './components/pages/VideoUploadPage';
// // import Home from './components/pages/HomePage'

// // Scroll to top on route change
// function ScrollToTop() {
//   const { pathname } = useLocation();

//   useEffect(() => {
//     window.scrollTo(0, 0);
//   }, [pathname]);

//   return null;
// }

// // Default export App component
// export default function App() {
//   return (
//     <Router>
//       <ScrollToTop />
//       <div style={styles.appContainer}>
//         <Navbar />
//         <div style={styles.contentContainer}>
//           <Routes>
//             <Route path="/" element={<HomePage />} />
//             <Route path="/upload" element={<VideoUploadPage />} />
//           </Routes>
//         </div>
//       </div>
//     </Router>
//   );
// }

// // Inline styles for App.js
// const styles = {
//   appContainer: {
//     minHeight: '100vh',
//     backgroundColor: '#f9fafb', // Equivalent to Tailwind's bg-gray-50
//   },
//   contentContainer: {
//     paddingTop: '64px', // Adjust for fixed navbar height
//   },
// };






// import React, { useEffect } from 'react';
// import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
// import Navbar from './components/Navbar';
// import HomePage from './components/pages/HomePage'; // Ensure this import is correct
// import VideoUploadPage from './components/pages/VideoUploadPage';

// function ScrollToTop() {
//   const { pathname } = useLocation();

//   useEffect(() => {
//     window.scrollTo(0, 0);
//   }, [pathname]);

//   return null;
// }

// export default function App() {
//   return (
//     <Router>
//       <ScrollToTop />
//       <div style={styles.appContainer}>
//         <Navbar />
//         <div style={styles.contentContainer}>
//           <Routes>
//             <Route path="/" element={<HomePage />} />
//             <Route path="/upload" element={<VideoUploadPage />} />
//           </Routes>
//         </div>
//       </div>
//     </Router>
//   );
// }

// const styles = {
//   appContainer: {
//     minHeight: '100vh',
//     backgroundColor: '#f9fafb',
//   },
//   contentContainer: {
//     paddingTop: '64px',
//   },
// };








import React, { useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom';
import Navbar from './components/Navbar';
import HomePage from './components/pages/HomePage';
import VideoUploadPage from './components/pages/VideoUploadPage';
import VideoUploadSection from './components/sections/VideoUploadSection';

// Scroll to top on route change
function ScrollToTop() {
  const { pathname } = useLocation();

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  return null;
}

// Default export App component
export default function App() {
  return (
    <Router>
      <ScrollToTop />
      <div style={styles.appContainer}>
        <Navbar />
        <div style={styles.contentContainer}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/upload" element={<VideoUploadPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

// Inline styles for App.js
const styles = {
  appContainer: {
    minHeight: '100vh',
    backgroundColor: '#f9fafb', // Equivalent to Tailwind's bg-gray-50
  },
  contentContainer: {
    paddingTop: '64px', // Adjust for fixed navbar height
  },
};