import template from './main-menu.pug'
import './main-menu.scss'

export default class MainMenu {
  constructor (node, data) {
    this.node = node
    this.node.innerHTML = template(data)
  }
}
