const { createProxyMiddleware } = require('http-proxy-middleware');

module.exports = function(app) {
  app.use(
    '/api/**',  // Match /api and all sub-paths
    createProxyMiddleware({
      target: 'http://localhost:8080',
      changeOrigin: true,
      pathRewrite: {
        '^/api/(.*)': '/api/$1'  // Explicitly keep /api prefix
      }
    })
  );
};
