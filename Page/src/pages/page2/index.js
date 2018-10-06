import data from './../../../data/pages/page2.json'
import MainMenu from './../../modules/mainMenu/main-menu'

document.title = data.title

// Components of the page
const main = document.querySelector('.main-container')

/* eslint-disable no-new */
const menuNode = document.createElement('DIV')
main.appendChild(menuNode)
new MainMenu(menuNode, data.components.menu1)

const menu2Node = document.createElement('DIV')
main.appendChild(menu2Node)
new MainMenu(menu2Node, data.components.menu2)
