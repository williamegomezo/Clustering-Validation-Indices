const path = require('path')
const merge = require('webpack-merge')
const common = require('./webpack.common.js')
const CopyWebpackPlugin = require('copy-webpack-plugin')
const UglifyJsPlugin = require('uglifyjs-webpack-plugin')

module.exports = merge(common, {
  devtool: 'inline-source-map',
  plugins: [
    new UglifyJsPlugin({
      sourceMap: true
    }),
    new CopyWebpackPlugin([
      {
        from: 'src/assets/',
        to: 'assets/',
        ignore: [ '*.scss' ]
      }
    ])
  ]
})
