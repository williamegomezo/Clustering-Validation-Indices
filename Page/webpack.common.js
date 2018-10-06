const path = require('path')
const CleanWebpackPlugin = require('clean-webpack-plugin')
const ExtractTextPlugin = require('extract-text-webpack-plugin')
const StyleLintPlugin = require('stylelint-webpack-plugin')
const HtmlWebpackPlugin = require('html-webpack-plugin')

const PATHS = {
  pug: __dirname + '/src/pages/'
}

function Pages(name) {
  return new HtmlWebpackPlugin({
    filename: name === 'index' ? 'index.html' : name + '/index.html',
    template: '!!pug-loader!' + PATHS.pug + 'layout.pug',
    inject: false,
    chunks: [name]
  })
}

module.exports = {
  entry: {
    index: ['./src/pages/layout.js', './src/pages/index/index.js']
  },
  output: {
    path: path.resolve(__dirname, '../docs'),
    filename: '[name]/index.js'
  },
  module: {
    rules: [
      {
        test: /\.scss$/,
        use: [{
          loader: 'style-loader'// creates style nodes from JS strings
        }, {
          loader: 'css-loader',
          options: {
            minimize: true,
            sourceMap: true
          }
          // translates CSS into CommonJS
        }, {
          loader: 'sass-loader',
          options: {
            sourceMap: true
          }
        }]
      },
      {
        enforce: 'pre',
        test: /\.js$/,
        loader: 'standard-loader',
        exclude: /(node_modules)/,
        options: {
          error: false,
          snazzy: true
        }
      },
      {
        test: /\.js$/,
        loader: 'babel-loader',
        query: {
          presets: ['es2015']
        }
      },
      {
        include: /\.pug/,
        use: [{loader: 'pug-loader'}]
      },
      {
        test: /\.(png|jp(e*)g|svg)$/,
        use: [{
          loader: 'file-loader',
          options: {
            name: 'assets/images/[name].[ext]'
          }
        }]
      }
    ]
  },
  plugins: [
    new ExtractTextPlugin({
      filename: 'styles.css',
      allChunks: true
    }),
    new StyleLintPlugin(),
    Pages('index')
  ]
}
